# Onboarding — read this first

## Update (2026-07-30): the fork below has been decided — read this first

The fork described later in this document (fix DeepONet vs. recenter on the
CNN) has been **decided: recenter on the CNN**, and gone further than
originally scoped. Since the fork section below was written, the
double-penalty investigation (`RESEARCH_LOG.md` items 24–35) landed a
confirmed, full-scale, 5-seed result: an AMSE training-objective loss
(adapted from Subich et al. 2025, ICML) closes the eddy-tracking
double-penalty gap that architecture change alone could not. This is now the
manuscript's central, novel contribution. **A full restructuring plan exists
at `MANUSCRIPT_RESTRUCTURE_PLAN.md` — read that before doing anything else
if you're picking up manuscript work.** It supersedes the "suggested opening
move" at the bottom of this file (that question — fix DeepONet or pivot — is
answered; don't re-ask it). The rest of this document (below) is left as
historical context for how the decision was reached, not as an open question.

## What this project is

A manuscript forecasting mesoscale eddies in the Agulhas Current using a
physics-informed DeepONet trained on GLORYS12V1 reanalysis
(`Agulhas_DeepONet_Manuscript.{md,tex,pdf}`). As currently written, it reports
four co-equal findings: (1) a well-tuned whole-domain DeepONet beats persistence
modestly at 1-day lead (+0.043 mean skill) but decays under multi-step rollout
and shows no eddy-tracking improvement; (2) divergence-free/geostrophic physics
constraints are statistically confirmed inert; (3) a patch-based DeepONet fixes
a resolution-scaling failure and was, until recently, the paper's strongest
grid-point result (+0.075); (4) weekly-aggregated forecasting shows *growing*
(not decaying) rollout skill, but only for two variables (`thetao`, `mlotst`),
confirmed via a climatology control and a split-boundary embargo check.

## THE SITUATION — why a big decision is on the table right now

An external review (11 issues, tracked in `MANUSCRIPT_ISSUES.md`) prompted
building a non-persistence learned baseline — a small U-Net (review Issue 3).
Result, now confirmed at full statistical rigor (5 seeds, formal paired t-test
against the DeepONet's own 5-seed numbers, `compare_cnn_vs_deeponet.py`):

- **Single-step:** DeepONet +0.0412±0.0061 vs. CNN +0.3797±0.0089, paired
  **t=-81.99**. The CNN beats the whole-domain DeepONet by ~9x, using 33x
  fewer parameters (434K vs. 14.2M).
- **Rollout:** CNN wins massively at 1d/5d/10d (t=-82, -36, -10), but at 20d
  the two are statistically indistinguishable (**t=+0.38**) — both collapse to
  negative skill under enough autoregressive rollout.
- The patch DeepONet's own headline (+0.075, previously called "the paper's
  strongest, most robust finding") is dwarfed by this plain CNN too.

This directly threatens the paper's premise: is DeepONet doing anything
architecturally distinctive for this task, or does "well-tuned local model +
persistence-residual skip + correct learning rate" just work regardless of
architecture? Diagnosis so far points at one specific, literature-consistent
explanation — not "DeepONet is bad" in general: the whole-domain branch is a
dense MLP reading the entire flattened grid, with no spatial locality or
translation-equivariance. Recent literature (Raonić et al. 2023) attributes
DeepONet/FNO's documented weak zero-shot resolution transfer to exactly this.

## Attempts to fix DeepONet so far (mixed — read before repeating)

1. **CNN-branch whole-domain DeepONet** (`train_agulhas_deeponet_cnnbranch.py`)
   — replace only the branch's flatten+MLP with a small conv encoder; trunk,
   persistence residual, and combination mechanism unchanged. Local screen:
   genuinely promising (+0.0589 at 700 days / 1500 undertrained iterations —
   already beats the *original* DeepONet's full-scale headline of +0.043).
   **The full-scale LR sweep (`cnnbranch_lr_sweep.slurm`) has not been run.
   This is the single most important open experiment.**
2. **DD-DeepONet** (`train_agulhas_deeponet_patch_dd.py`) — soft
   interface-consistency penalty between overlapping patch tiles. Local
   screen: flat/inconclusive (weight 0/0.01/0.1/1.0 all landed within
   +0.054–0.058, no trend). Held at the user's request, not pursued further.
3. **Patch + CNN branch combined** (`train_agulhas_deeponet_patch_cnnbranch.py`)
   — CNN encoder inside each tile. Local, matched-iteration control: *worse*
   than plain patch DeepONet (+0.0415 vs. +0.0524), not just noise — val loss
   was worse too. Likely explanation: a 20×20 tile flattened is only 2,400
   dimensions, small enough that a dense MLP already handles it fine — the
   branch-locality problem seems to be whole-domain-scale specifically, not
   tile-scale. Caveat: parameter count wasn't controlled between the two
   configs (206K vs. 966K), so this isn't fully conclusive.

## The fork the user is deciding between

- **(a) Fix DeepONet.** Run the CNN-branch full-scale sweep (item 1 above); if
  it closes most of the gap, keep the DeepONet-centered framing with a real,
  evidence-based architectural fix rather than a guess.
- **(b) Recenter the paper on the CNN**, using the eddy-tracking
  grid-skill/eddy-skill dissociation as the central finding instead of
  DeepONet's architecture. This finding gets *more* compelling with the CNN
  as backbone, not less: a model with ~40% grid-point skill still showing
  zero eddy-tracking improvement is a much sharper demonstration of the
  double-penalty problem (Ebert 2008; Gilleland et al. 2009) than the
  original +4%-skill framing ever was. The physics-circularity finding
  (Issue 1) survives either way — it's architecture-agnostic.

Standing recommendation (given to the user previously, not yet acted on to
completion): try (a) first, since it's cheap relative to a full pivot and
answers "is DeepONet salvageable" with evidence instead of speculation. If
you're picking this back up, **`sbatch cnnbranch_lr_sweep.slurm` is probably
the highest-value next action** before committing to anything bigger.

## Where to find more detail

- `MANUSCRIPT_ISSUES.md` — 11 external-review issues, severity-tiered, each
  with the reviewer's argument, what fixing it requires, and current status.
  Issue 3 (CNN baseline) and Issue 4 (discretization invariance) are the ones
  entangled with the decision above.
- `RESEARCH_LOG.md` — full chronological log, numbered entries. The most
  recent ones (search for `2026-07-28` / `2026-07-29`) cover everything above
  in exact technical detail: commands run, exact numbers, code design
  rationale.
- The manuscript has **not** been updated to reflect any of the CNN-baseline
  findings above. That's deliberate — this study's own standard is not to
  touch the manuscript until an open experiment/decision affecting it is
  settled.

## Key files

| File | What it is |
|---|---|
| `Agulhas_DeepONet_Manuscript.{md,tex,pdf}` | The manuscript (`.md` is source of truth; `.tex` is a hand-mirrored twin; rebuild PDF with `tectonic Agulhas_DeepONet_Manuscript.tex`) |
| `RESEARCH_LOG.md` | Full chronological research log (append-only) |
| `MANUSCRIPT_ISSUES.md` | External-review issue tracker |
| `train_agulhas_deeponet_prototype.py` | Main whole-domain DeepONet trainer (Table 1/3's model) |
| `train_agulhas_deeponet_patch.py` | Patch-based DeepONet (Table 7's model, dense branch) |
| `train_cnn_baseline.py` | Non-persistence CNN/U-Net baseline — the architecture currently winning |
| `train_agulhas_deeponet_cnnbranch.py` | CNN-branch DeepONet — the "fix DeepONet" attempt; promising locally, **full-scale untested** |
| `train_agulhas_deeponet_patch_cnnbranch.py` | Patch + CNN branch combined — negative local result |
| `train_agulhas_deeponet_patch_dd.py` | DD-DeepONet interface-consistency variant — flat local result |
| `compare_cnn_vs_deeponet.py` | Formal paired-t comparison script (works on any two seed-swept result directories) |
| `aggregate_seed_sweep.py`, `extract_rollout_rmse.py` | Generic multi-seed aggregation tools; work on any `<tag>_seed<N>/{metrics.json,rollout.npz}` directory unmodified |
| `cnnbranch_lr_sweep.slurm` | **Not yet run** — the highest-priority pending cluster job |
| `HANDOFF.md`, `RESULTS.md`, `TECHNICAL_WALKTHROUGH.md` | **Stale, dated 2026-07-05.** Pre-manuscript-restructuring, pre-CNN-discovery planning docs. Background reading only — not authoritative, and `RESULTS.md` references a `Research Proposal.md` that no longer exists in this directory. |

## Suggested opening move

Don't re-derive any of the above — it's already established and logged. The
live question is strategic, not technical: given the CNN-branch DeepONet's
promising-but-unconfirmed local result, does the user want to (1) run the
full-scale sweep and decide from real numbers, or (2) skip straight to the
pivot toward a CNN-centered, double-penalty-focused paper? Ask, don't assume
which one they want — this is exactly the kind of call that's genuinely
theirs to make, not something to infer from the code or prior conversation.
