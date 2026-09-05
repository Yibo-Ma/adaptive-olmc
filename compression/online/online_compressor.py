"""OnlineCompressor: LoRA-adaptive chunked compression.

Compress chunk-by-chunk; after every full interval of ``train_interval`` chunks,
fine-tune a LoRA adapter on those (just-coded) chunks.  Weights are constant
within an interval, so its chunks share one forward batch.  The decoder replays
the identical init + training schedule on the *decoded* chunks, reaching a
bit-identical model state at every interval boundary, so no adapter is ever
transmitted.

The partial trailing interval is coded but not trained (mirrored on both ends).
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from compression.online.adjacency_plan import plan_adjacency
from compression.online.backends.base import ChunkUnit, OnlineBackend
from compression.online.base import _ChunkedCompressor
from compression.online.config import OnlineLearningConfig
from compression.online.trainer import OnlineTrainer, build_optimizer


class OnlineCompressor(_ChunkedCompressor):

    ROLE = "online"

    def __init__(
        self, backend: OnlineBackend, device: torch.device, cfg: OnlineLearningConfig,
        shuffle_seed: Optional[int] = None, measure_gk: bool = False,
        branch_lrs: Optional[List[float]] = None, branch_ref_lr: Optional[float] = None,
        adjacency_probe: Optional[Dict] = None, domain_blocks: Optional[List[Dict]] = None,
        ctx_tokens: int = 0,
    ) -> None:
        super().__init__(backend, device, shuffle_seed=shuffle_seed,
                         ctx_tokens=ctx_tokens)
        # The probes measure through measure_interval_bits, which mirrors the
        # BOS-prefixed coding path only — pairing them with cross-chunk context would
        # report bits the coder never bills.  Refuse rather than misreport.
        if ctx_tokens and (measure_gk or branch_lrs or adjacency_probe):
            raise ValueError(
                "cross-chunk context (ctx_tokens>0) cannot be combined with the "
                "g_k / branch / adjacency probes: measure_interval_bits still "
                "mirrors the BOS-prefixed path. Run them separately.")
        self.cfg = cfg
        self.optimizer = None
        self.trainer = None
        # g_k instrument (see _record_gk).  None disables it and keeps the coding
        # path byte-identical; a list collects one prequential record per training
        # boundary.  Encoder-side only — decoding never needs it.
        self.gk_records: Optional[List[Dict]] = [] if measure_gk else None
        # Same-state branching instrument (see _record_branch): at each boundary,
        # score the next interval under several candidate learning rates from the
        # identical parent state, then advance the MAIN trajectory by branch_ref_lr.
        # None disables it (coding path byte-identical); mutually exclusive with gk.
        self.branch_lrs: Optional[List[float]] = branch_lrs
        self.branch_ref_lr: float = (
            branch_ref_lr if branch_ref_lr is not None else cfg.learning_rate)
        self.branch_records: Optional[List[Dict]] = [] if branch_lrs else None
        # V1-A adjacency probe (see _record_adjacency): at sampled boundaries score the
        # true future window and matched non-adjacent controls under the pre/post-update
        # model.  The main trajectory is plain OSOA (byte-identical); only extra frozen
        # forwards are added.  dict{h_max,k_controls,n_target,seed} enables it; None off.
        self.adjacency_probe: Optional[Dict] = adjacency_probe
        self.domain_blocks: Optional[List[Dict]] = domain_blocks
        self.adjacency_records: Optional[List[Dict]] = [] if adjacency_probe else None
        self._adj_plan: Dict[int, List[int]] = {}

    def _settings(self) -> Dict:
        return asdict(self.cfg)

    def setup(self) -> None:
        self.backend.load_backbone()
        self.backend.attach_lora(self.cfg)          # before make_compressor: wrap PEFT model
        self.compressor = self.backend.make_compressor()
        self.optimizer = build_optimizer(self.backend.model, self.cfg)
        self.trainer = OnlineTrainer(self.cfg, self.device)

    # ------------------------------------------------------------------

    def _train(self, train_chunks: List[ChunkUnit], phase: int) -> None:
        windows = self.backend.build_training_windows(train_chunks, self.cfg)
        self.trainer.train_phase(
            self.backend.model, self.optimizer, windows, phase, self.backend.compute_loss,
        )

    def compress(self, raw: Any, framing: bytes = b"") -> bytes:
        chunks = self._prepare_chunks(raw)
        total_ob = self.backend.raw_size_bytes(raw)
        # Materialised so a training boundary can look ahead to the next interval
        # (the g_k probe).  Same grouping the decoder replays.
        intervals = list(self._iter_intervals(chunks, self.cfg.train_interval))
        if self.adjacency_records is not None:
            self._build_adjacency_plan(intervals)

        all_cds = []
        seen: List[ChunkUnit] = []
        phase = 0
        for k, (group, is_tail) in enumerate(intervals):
            all_cds.extend(self.backend.encode_interval(
                self.compressor, group, ctx_ids=self.ctx.tail()))
            self.ctx.extend(c.token_ids for c in group)
            seen.extend(group)
            if is_tail:
                continue
            train_chunks = group if self.cfg.train_on_recent_only else seen
            nxt = intervals[k + 1][0] if k + 1 < len(intervals) else None
            if nxt is None:
                self._train(train_chunks, phase)          # last full interval: advance only
            elif self.adjacency_records is not None:
                if k in self._adj_plan:
                    self._record_adjacency(phase, k, intervals, self._adj_plan[k], train_chunks)
                else:
                    self._train(train_chunks, phase)
            elif self.branch_records is not None:
                self._record_branch(phase, group, nxt, train_chunks)
            elif self.gk_records is not None:
                self._record_gk(phase, group, nxt, train_chunks)
            else:
                self._train(train_chunks, phase)
            phase += 1

        return self._assemble_archive(self.ROLE, all_cds, total_ob, framing)

    # ------------------------------------------------------------------
    # g_k instrument (encoder-side measurement; off unless measure_gk=True)
    # ------------------------------------------------------------------

    def _measure_interval(self, chunks: List[ChunkUnit]):
        """(bits, tokens, orig_bytes) for one interval at the current model state."""
        bits = sum(self.backend.measure_interval_bits(self.compressor, chunks))
        tokens = sum(len(c.token_ids) for c in chunks)
        orig_bytes = self.backend.raw_size_bytes(self.backend.from_chunks(chunks))
        return bits, tokens, orig_bytes

    def _measure_interval_base(self, chunks: List[ChunkUnit]) -> float:
        """Total interval bits under the FROZEN base model = L(I|S_0), by disabling
        the LoRA adapter (equivalently S_0, since LoRA-B is zero at init).  The base
        term of regret-vs-base = L(I_k|S_k) − L(I_k|S_0): >0 means the adapted model
        is already worse than doing nothing = cumulative overfitting/drift, which g_k
        (a local, adjacent-state quantity) cannot see."""
        with self.backend.model.disable_adapter():
            return sum(self.backend.measure_interval_bits(self.compressor, chunks))

    def _record_gk(
        self, phase: int, curr: List[ChunkUnit], nxt: List[ChunkUnit],
        train_chunks: List[ChunkUnit],
    ) -> None:
        """Measure the prequential g_k and Δ_in around one training step.

        g_k = L(next | S_k) − L(next | S_{k+1}) isolates the marginal effect of
        *this* update on the *next* interval; Δ_in = L(curr | S_k) − L(curr | S_{k+1})
        its effect on the interval it trained on.  We store only raw code lengths
        (bits) plus each interval's token and original-byte counts, so the consumer
        (evaluation/gk_curve.py) can form the differences and normalise them any way
        it likes (Δbpb, relative %, bits/token) — the model run stays the sole
        producer of ground-truth numbers.  ``curr.bits_base`` = L(curr|S_0) is added
        for the regret-vs-base signal (see _measure_interval_base).

        The pre-update reads (incl. the adapter-disabled base read) see state S_k;
        ``_train`` advances it to S_{k+1} and reseeds the global RNG at entry
        (utils.determinism.set_seed), so these inference-only forwards cannot perturb
        the training trajectory — losslessness is preserved and the archive is
        byte-identical to an un-instrumented run.
        """
        curr_pre, curr_tok, curr_by = self._measure_interval(curr)
        curr_base = self._measure_interval_base(curr)      # L(curr|S_0), for regret-vs-base
        next_pre, next_tok, next_by = self._measure_interval(nxt)
        self._train(train_chunks, phase)
        curr_post, _, _ = self._measure_interval(curr)
        next_post, _, _ = self._measure_interval(nxt)
        self.gk_records.append({
            "phase": phase,
            "curr": {"tokens": curr_tok, "bytes": curr_by,
                     "bits_pre": curr_pre, "bits_post": curr_post, "bits_base": curr_base},
            "next": {"tokens": next_tok, "bytes": next_by,
                     "bits_pre": next_pre, "bits_post": next_post},
        })

    # ------------------------------------------------------------------
    # Same-state branching instrument (encoder-side; off unless branch_lrs set)
    # ------------------------------------------------------------------

    def _snapshot_state(self):
        """Clone the branchable state at the current boundary: LoRA (trainable)
        params + full optimizer (Adam moment) state.  Restoring returns the model
        and optimizer bit-exactly to S_k, so every candidate branches from an
        identical parent.  The global RNG is deliberately NOT snapshotted:
        ``train_phase`` reseeds deterministically per phase at entry, so a
        candidate's (or the advance's) training is independent of whatever RNG the
        sibling branches consumed — the same property that makes _record_gk safe."""
        params = {n: p.detach().clone()
                  for n, p in self.backend.model.named_parameters() if p.requires_grad}
        opt = copy.deepcopy(self.optimizer.state_dict())
        return params, opt

    def _restore_state(self, snapshot) -> None:
        params, opt = snapshot
        with torch.no_grad():
            for n, p in self.backend.model.named_parameters():
                if p.requires_grad:
                    p.copy_(params[n])
        self.optimizer.load_state_dict(opt)

    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _record_branch(
        self, phase: int, curr: List[ChunkUnit], nxt: List[ChunkUnit],
        train_chunks: List[ChunkUnit],
    ) -> None:
        """Same-state one-step branching (the dynamic-strength probe).

        From the parent state S_k, apply each candidate learning rate, score the
        NEXT interval (and the just-coded one) under the resulting state, then
        discard every branch and advance the MAIN trajectory by ``branch_ref_lr``.
        Every candidate is restored to the identical parent (params + optimizer
        moments) and trains under the identical per-phase seed (train_phase reseeds
        at entry), so the ONLY difference between candidates is the learning rate —
        a clean counterfactual, exactly what "is the best strength different per
        interval?" needs.

        ``lr == 0`` is the *skip* (no-update) candidate: scored at S_k with no
        training, so its bits ARE L(·|S_k) — the hold baseline the consumer forms
        g_k and Δ_in against.  Only raw code lengths (bits, the measure_interval_bits
        NLL twin) plus token/byte counts are stored; evaluation/branch_curve.py forms
        best-action, the local oracle, the switch stats and near-ties.

        Determinism: the advance restores S_k and trains at ``branch_ref_lr`` exactly
        as a non-branching fixed-lr run would, so the coded trajectory — and thus the
        archive — is byte-identical to a plain run at that lr (the branches only read)."""
        curr_tok = sum(len(c.token_ids) for c in curr)
        curr_by = self.backend.raw_size_bytes(self.backend.from_chunks(curr))
        next_tok = sum(len(c.token_ids) for c in nxt)
        next_by = self.backend.raw_size_bytes(self.backend.from_chunks(nxt))

        # Frozen-base (Static) reference on both intervals = L(·|S_0), via disable_adapter.
        curr_base = self._measure_interval_base(curr)
        next_base = self._measure_interval_base(nxt)

        snapshot = self._snapshot_state()
        advance_state = None       # reused reference post-state (saves one train/boundary)
        cands: List[Dict] = []
        for lr in self.branch_lrs:
            self._restore_state(snapshot)        # every candidate branches from S_k
            if lr == 0.0:                        # skip = no update
                curr_bits = self._measure_interval(curr)[0]
                next_bits = self._measure_interval(nxt)[0]
            else:
                self._set_lr(lr)
                self._train(train_chunks, phase)
                curr_bits = self._measure_interval(curr)[0]
                next_bits = self._measure_interval(nxt)[0]
                # The reference candidate IS the main advance (same parent, lr and
                # per-phase seed): snapshot its post-state and reuse it below instead
                # of training a redundant fourth time — bit-identical, ~25% cheaper.
                if lr == self.branch_ref_lr:
                    advance_state = self._snapshot_state()
            cands.append({
                "lr": lr, "curr_bits": curr_bits, "next_bits": next_bits,
                "nonfinite": not (math.isfinite(curr_bits) and math.isfinite(next_bits)),
            })

        # Advance the main trajectory to S_{k+1} at the reference lr.  Reuse the
        # reference candidate's post-state when it was one of the branches; else train.
        if advance_state is not None:
            self._restore_state(advance_state)
        else:
            self._restore_state(snapshot)
            self._set_lr(self.branch_ref_lr)
            self._train(train_chunks, phase)

        self.branch_records.append({
            "phase": phase,
            "ref_lr": self.branch_ref_lr,
            "curr": {"tokens": curr_tok, "bytes": curr_by, "bits_base": curr_base},
            "next": {"tokens": next_tok, "bytes": next_by, "bits_base": next_base},
            "candidates": cands,
        })

    # ------------------------------------------------------------------
    # V1-A adjacency probe (encoder-side; off unless adjacency_probe set)
    # ------------------------------------------------------------------

    def _domain_of(self, byte_offset: int) -> object:
        """Domain label for a stream byte offset (via domain_blocks, else one stream)."""
        if not self.domain_blocks:
            return "stream"
        for blk in self.domain_blocks:
            if blk["byte_start"] <= byte_offset < blk["byte_end"]:
                return blk["dataset"]
        return self.domain_blocks[-1]["dataset"]

    def _build_adjacency_plan(self, intervals) -> None:
        """Pre-pass: per-interval frozen-base (Static) NLL and domain label, then pick
        the sampled boundaries + matched controls (pure plan_adjacency).  Static NLL is
        read under disable_adapter so difficulty matching is adapter-independent."""
        groups = [g for g, _ in intervals]
        with self.backend.model.disable_adapter():
            static_nll = [float(sum(self.backend.measure_interval_bits(self.compressor, g)))
                          for g in groups]
        domains, cum = [], 0
        for g in groups:
            domains.append(self._domain_of(cum))
            cum += self.backend.raw_size_bytes(self.backend.from_chunks(g))
        p = self.adjacency_probe
        self._adj_static = static_nll
        self._adj_domains = domains
        self._adj_plan = dict(plan_adjacency(
            domains, static_nll, p["h_max"], p["k_controls"], p["n_target"], p["seed"]))

    def _record_adjacency(
        self, phase: int, k: int, intervals, controls: List[int],
        train_chunks: List[ChunkUnit],
    ) -> None:
        """Score the true future window [k+1, k+h_max] and each matched control window
        under the pre-update model θ_t, apply the normal OSOA update (θ_t -> θ_t^+), then
        score the same windows again under θ_t^+.  All bits are the measure_interval_bits
        NLL twin (framing-free), stored as per-horizon cumulative prefixes so H=1..h_max
        come from one pass.  The update IS the plain-OSOA advance and the scorings are
        no-grad reads, so the coded trajectory is byte-identical to a plain run."""
        h = self.adjacency_probe["h_max"]

        def score_prefixes(start: int) -> List[float]:
            cum, out = 0.0, []
            for i in range(h):
                cum += float(sum(self.backend.measure_interval_bits(
                    self.compressor, intervals[start + i][0])))
                out.append(cum)
            return out

        def mean_prefix(mat: List[List[float]]) -> List[float]:
            return [sum(row[i] for row in mat) / len(mat) for i in range(h)]

        curr = intervals[k][0]
        curr_pre = float(sum(self.backend.measure_interval_bits(self.compressor, curr)))
        adj_pre = score_prefixes(k + 1)
        ctrl_pre = [score_prefixes(j) for j in controls]

        self._train(train_chunks, phase)                      # plain-OSOA advance to θ_t^+

        curr_post = float(sum(self.backend.measure_interval_bits(self.compressor, curr)))
        adj_post = score_prefixes(k + 1)
        ctrl_post = [score_prefixes(j) for j in controls]

        self.adjacency_records.append({
            "phase": phase, "boundary_k": k,
            "domain": str(self._adj_domains[k + 1]),
            "horizons": list(range(1, h + 1)),
            "future_static_bits": float(sum(self._adj_static[k + 1:k + 1 + h])),
            "control_starts": controls,
            "curr_pre_bits": curr_pre, "curr_post_bits": curr_post,
            "adj_pre": adj_pre, "adj_post": adj_post,
            "ctrl_pre": mean_prefix(ctrl_pre), "ctrl_post": mean_prefix(ctrl_post),
        })

    def decompress(self, archive_bytes: bytes) -> Any:
        total_ob, framing, cds = self._open_archive(self.ROLE, archive_bytes)

        decoded: List[ChunkUnit] = []
        seen: List[ChunkUnit] = []
        phase = 0
        for group, is_tail in self._iter_intervals(cds, self.cfg.train_interval):
            chunk_units = self.backend.decode_interval(
                self.compressor, group, ctx_ids=self.ctx.tail())
            self.ctx.extend(c.token_ids for c in chunk_units)
            decoded.extend(chunk_units)
            seen.extend(chunk_units)
            if not is_tail:
                train_chunks = chunk_units if self.cfg.train_on_recent_only else seen
                self._train(train_chunks, phase)
                phase += 1

        return self._finalize(self._restore_order(decoded), total_ob, framing)
