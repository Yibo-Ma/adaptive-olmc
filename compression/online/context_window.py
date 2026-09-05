"""ContextWindow: the rolling tail of already-coded tokens used as cross-chunk context.

By default the coder starts every chunk from a single BOS/dummy token, so no
information crosses a chunk boundary except through the adapted weights.  This
window optionally carries the last ``ctx_tokens`` already-coded token ids forward
so a chunk is predicted conditioned on the text that preceded it.

Both endpoints feed the window the *same* chunks in the *same* coding order (the
encoder as it codes them, the decoder as it decodes them), so the tail is
identical on both sides and costs zero transmitted bits — the same replay
argument that licenses the LoRA updates themselves.

Pure logic (no torch) so the trimming contract can be unit-tested directly.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


class ContextWindow:

    def __init__(self, ctx_tokens: int = 0) -> None:
        self.ctx_tokens = int(ctx_tokens or 0)
        self._buf: List[int] = []

    @property
    def enabled(self) -> bool:
        return self.ctx_tokens > 0

    def tail(self) -> Optional[List[int]]:
        """Context to prepend to the next interval, or None for the BOS default.

        None (not an empty list) at the very start, so the first interval takes the
        exact same code path as a run without cross-chunk context.
        """
        if not self.enabled or not self._buf:
            return None
        return self._buf[-self.ctx_tokens:]

    def extend(self, token_id_lists: Iterable[Sequence[int]]) -> None:
        """Append an interval's chunks (in coding order) and trim to the window."""
        if not self.enabled:
            return
        for ids in token_id_lists:
            self._buf.extend(ids)
        if len(self._buf) > self.ctx_tokens:
            del self._buf[:len(self._buf) - self.ctx_tokens]
