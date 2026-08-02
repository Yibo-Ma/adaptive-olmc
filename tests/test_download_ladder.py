"""Pure-logic tests for the download fallback ladder in scripts/download_models.py.

No network: checks the (endpoint, hf_transfer) attempt ordering and, crucially, that a
plain-HTTP attempt (hf_transfer OFF) is always the last resort — so a silent hf_transfer
0-byte failure self-heals in any environment.

    python tests/test_download_ladder.py        # plain script
    pytest tests/test_download_ladder.py         # if pytest is installed
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import download_models as dm  # noqa: E402

MIRROR, HFCO = "https://hf-mirror.com", "https://huggingface.co"


def test_prefers_fast_then_plain_per_endpoint():
    ladder = dm._attempt_ladder([MIRROR, HFCO], use_aria=False, prefer_fast=True)
    assert ladder == [(MIRROR, True), (MIRROR, False), (HFCO, True), (HFCO, False)]


def test_respects_explicit_opt_out():
    # user set HF_HUB_ENABLE_HF_TRANSFER=0 -> never attempt the fast path
    ladder = dm._attempt_ladder([HFCO], use_aria=False, prefer_fast=False)
    assert ladder == [(HFCO, False)]


def test_aria_ignores_transfer():
    ladder = dm._attempt_ladder([MIRROR, HFCO], use_aria=True, prefer_fast=True)
    assert ladder == [(MIRROR, None), (HFCO, None)]


def test_last_resort_is_always_plain_http():
    # whatever the preference or endpoint set, the final HF attempt disables hf_transfer
    for pref in (True, False):
        ladder = dm._attempt_ladder([MIRROR, HFCO], use_aria=False, prefer_fast=pref)
        assert ladder[-1][1] is False


def main() -> int:
    tests = [test_prefers_fast_then_plain_per_endpoint, test_respects_explicit_opt_out,
             test_aria_ignores_transfer, test_last_resort_is_always_plain_http]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK  {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
