#!/usr/bin/env python3
"""Regression test for the control-pairing bug (RESEARCH_LOG.md item 39,
manuscript Sec. 4.2): "every AMSE seed compared against a single, arbitrarily
chosen control seed rather than its own seed-matched control," which changed
the recall-significance count from 5/10 to 10/10 once fixed. `eddy_stat_test.py`
`--mode a-vs-b` takes two arbitrary file paths with no built-in check that
they're seed-matched -- exactly the interface that would let this happen
silently again. This test verifies the guard added to `eddy_stat_test.py`
(Issue 32b, a formal review's request for systematic auditing beyond the one
eddy-detection regression test) actually blocks a mismatched-seed comparison
and does not block a matched one.

Fast unit test, no real data/training/cluster needed:
    python3 test_control_pairing_regression.py
"""
import subprocess
import sys
import tempfile
import os

sys.path.insert(0, os.path.dirname(__file__))
from eddy_stat_test import extract_seed  # noqa: E402


def make_dummy_result(path, days=(0, 1, 2)):
    import json
    per_day = [{"day": d, "n_true_anti": 1, "n_true_cyc": 1,
                "model_dists": [10.0], "model_n_matched": 1, "model_n_other": 1,
                "persist_dists": [12.0], "persist_n_matched": 1, "persist_n_other": 1}
               for d in days]
    json.dump({"summary": {}, "per_day": per_day}, open(path, "w"))


def main():
    failures = []

    # --- Unit-level: extract_seed() itself ---
    cases = [
        ("amse_w0.01_seed2026.json", 2026),
        ("cnn_seed2026.json", 2026),
        ("cnn_seed7.json", 7),
        ("results_r6_mplfix/amse_w0.01_seed123.json", 123),
        ("no_seed_here.json", None),
        ("amse_w0.01_seed-5.json", -5),
    ]
    for fname, expected in cases:
        got = extract_seed(fname)
        if got != expected:
            failures.append(f"extract_seed({fname!r}) = {got!r}, expected {expected!r}")
        else:
            print(f"  extract_seed({fname!r}) = {got!r}  OK")

    # --- Integration-level: the actual CLI guard, via subprocess so we test
    # exactly what a real invocation would see, not just the importable function ---
    with tempfile.TemporaryDirectory() as tmp:
        a_matched = os.path.join(tmp, "amse_w0.01_seed2026.json")
        b_matched = os.path.join(tmp, "cnn_seed2026.json")
        b_mismatched = os.path.join(tmp, "cnn_seed7.json")
        make_dummy_result(a_matched)
        make_dummy_result(b_matched)
        make_dummy_result(b_mismatched)

        script = os.path.join(os.path.dirname(__file__), "eddy_stat_test.py")
        py = sys.executable

        print("\n  Testing mismatched-seed CLI call (should refuse)...")
        r = subprocess.run([py, script, "--a", a_matched, "--b", b_mismatched, "--mode", "a-vs-b"],
                            capture_output=True, text=True)
        if r.returncode == 0 or "REFUSING TO RUN" not in (r.stdout + r.stderr):
            failures.append(f"mismatched-seed call did NOT refuse (returncode={r.returncode}); "
                             f"stdout/stderr: {(r.stdout + r.stderr)[:300]}")
        else:
            print("    OK: refused as expected.")

        print("  Testing mismatched-seed CLI call with --allow-seed-mismatch (should proceed)...")
        r = subprocess.run([py, script, "--a", a_matched, "--b", b_mismatched, "--mode", "a-vs-b",
                             "--allow-seed-mismatch"], capture_output=True, text=True)
        if "REFUSING TO RUN" in (r.stdout + r.stderr):
            failures.append("mismatched-seed call with --allow-seed-mismatch still refused "
                             "(the override flag doesn't work)")
        else:
            print("    OK: proceeded as expected (override respected).")

        print("  Testing matched-seed CLI call (should proceed, no refusal)...")
        r = subprocess.run([py, script, "--a", a_matched, "--b", b_matched, "--mode", "a-vs-b"],
                            capture_output=True, text=True)
        if "REFUSING TO RUN" in (r.stdout + r.stderr):
            failures.append(f"matched-seed call was incorrectly refused; "
                             f"stdout/stderr: {(r.stdout + r.stderr)[:300]}")
        else:
            print("    OK: proceeded as expected (no false positive).")

    print()
    if failures:
        print(f"FAIL: {len(failures)} issue(s):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("PASS: control-pairing guard correctly blocks mismatched seeds, "
              "respects the explicit override, and does not false-positive on matched seeds.")
        sys.exit(0)


if __name__ == "__main__":
    main()
