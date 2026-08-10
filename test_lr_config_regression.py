#!/usr/bin/env python3
"""Regression test for the exact class of bug in RESEARCH_LOG.md item 33 /
manuscript Sec. 4.4: a CNN-training SLURM script silently defaulting to the
DeepONet's own learning rate (3e-4) instead of the CNN's established-correct
rate (1e-3), copied from the wrong template. That bug affected "every
full-scale AMSE result" before being caught by accident (comparing against
an independently-known-good baseline), and was STILL LIVE in
`cnn_seed_sweep.slurm` as of 2026-08-05 -- discovered only while building
this test (Issue 32b, a formal review's request for systematic auditing
beyond the eddy-detection regression test in test_synthetic_ground_truth.py).

This is a static check, not a training run: it parses every SLURM script
that invokes a CNN-training script (`train_cnn_baseline*.py`) and asserts
the *effective* learning rate default resolves to the established-correct
CNN rate, unless the script is on the explicit EXEMPT list below (and only
for a stated, checked reason -- "sweeps multiple rates by design" is the
only accepted reason, not "predates the fix").

Two ways a script can go wrong, both checked:
  1. `LR="${CNN_LR:-X}"` pattern with the wrong fallback X.
  2. A hardcoded `--learning-rate X` with the wrong X and no override variable.

Run locally, no cluster/GPU needed:
    python3 test_lr_config_regression.py
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
ESTABLISHED_CORRECT_CNN_LR = "1e-3"

# Scripts explicitly exempt from the single-default check, with the specific,
# checked reason -- not a place to add a script just to silence a failure.
EXEMPT = {
    "cnn_lr_sweep.slurm": "sweeps LRS=\"1e-3 3e-4 1e-4\" by design, no single "
                           "default to check; verified 1e-3 is included in the sweep list.",
}


def find_cnn_training_slurm_scripts():
    scripts = []
    for f in REPO_ROOT.glob("*.slurm"):
        text = f.read_text()
        if re.search(r"train_cnn_baseline\w*\.py", text):
            scripts.append(f)
    return scripts


def extract_effective_lr(text):
    """Returns list of (source_line, effective_lr) for every LR-determining
    construct found. A script can have more than one (e.g. a variable
    default AND a hardcoded flag elsewhere -- both are checked)."""
    findings = []
    for m in re.finditer(r'^\s*LR="\$\{CNN_LR:-([^}]+)\}"', text, re.MULTILINE):
        findings.append((m.group(0).strip(), m.group(1)))
    for m in re.finditer(r'--learning-rate\s+([0-9.eE+-]+)(?!["\'\}])', text):
        # Skip occurrences that are clearly a shell variable reference like
        # --learning-rate "${LR}" -- those resolve via the LR= line above,
        # already captured, not a second independent hardcoded value.
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start:text.find("\n", m.start())]
        if "${LR}" in line or "${LR:" in line or "\"${LR}\"" in line:
            continue
        findings.append((line.strip(), m.group(1)))
    return findings


def main():
    scripts = find_cnn_training_slurm_scripts()
    if not scripts:
        print("FAIL: found zero SLURM scripts invoking train_cnn_baseline*.py "
              "-- the glob/grep pattern itself may be broken, not a clean pass.")
        sys.exit(1)

    print(f"Checking {len(scripts)} SLURM script(s) that invoke a CNN training script...")
    failures = []
    for script in sorted(scripts):
        name = script.name
        if name in EXEMPT:
            print(f"  {name}: EXEMPT ({EXEMPT[name]})")
            continue
        text = script.read_text()
        findings = extract_effective_lr(text)
        if not findings:
            failures.append((name, "no LR default or hardcoded --learning-rate found "
                                    "-- cannot verify this script uses the correct rate"))
            continue
        for source_line, lr in findings:
            if lr != ESTABLISHED_CORRECT_CNN_LR:
                failures.append((name, f"resolves to lr={lr}, expected {ESTABLISHED_CORRECT_CNN_LR} "
                                        f"-- from: {source_line}"))
            else:
                print(f"  {name}: OK (lr={lr})")

    print()
    if failures:
        print(f"FAIL: {len(failures)} issue(s) found:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("PASS: every non-exempt CNN-training SLURM script resolves to the "
              f"established-correct rate ({ESTABLISHED_CORRECT_CNN_LR}).")
        sys.exit(0)


if __name__ == "__main__":
    main()
