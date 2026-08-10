# Manuscript Restructure Plan — CNN Eddy Forecasting as the Main Result

**Status:** Decision made (2026-07-30), not yet executed. This document is the
handoff artifact for whichever session does the actual rewrite. Read this
after `ONBOARDING.md`, `MANUSCRIPT_ISSUES.md`, and `RESEARCH_LOG.md` — it
assumes familiarity with all three and doesn't re-derive their content, only
tells you what to do with it.

---

## Direction

The manuscript is being completely reorganized around a three-part finding
that did not exist when the original DeepONet-centered draft was written:
(1) a CNN beats the whole-domain DeepONet on this task by roughly an order of
magnitude, confirmed at 5 seeds, and the literature already explains why
(DeepONet's namesake discretization invariance never held on the input/branch
side); (2) even a dramatically more skillful CNN doesn't reliably translate
grid-point accuracy into eddy-level forecast quality — architecture alone
fixes eddy *position* error but not detection *recall*, the double-penalty
problem (Ebert 2008); (3) that gap is fixable, but only at the
training-objective level — an AMSE (amplitude-adjusted MSE) loss adapted from
Subich et al. (2025, ICML), built originally to fix an analogous smoothing
problem in global spherical weather forecasting, closes both the recall and
position-error gaps simultaneously when ported to this regional ocean domain
via a 2D FFT. All three legs are now confirmed at this project's own
gold-standard rigor tier (5 seeds, full 1993–2021 scale, paired
significance testing) — see `RESEARCH_LOG.md` items 20–21, 23, 27–35 for the
complete evidentiary trail.

This replaces the old four-co-equal-findings structure (physics ablation,
patch-resolution scaling, eddy-tracking null result, weekly-aggregation
timescale finding) with a single, tight narrative arc. The physics-ablation,
patch-scaling, and weekly-aggregation findings are being cut or shrunk, not
because they were wrong, but because none of them have any demonstrated
connection to the CNN story — they were run on the DeepONet and never
retested on the architecture the paper now centers on, and keeping them
would recreate exactly the "four loosely related studies stapled together"
problem the external reviewer already flagged (Issue 10). Nothing is lost
permanently: the DeepONet architecture work is fully documented in
`RESEARCH_LOG.md` and remains a well-scoped candidate for a second,
shorter paper after this one is done, if there's appetite for it later.

The guiding principle for the rewrite: every claim in the new draft should be
able to point to a specific, already-completed, multi-seed, full-scale
result. That's true of the whole core spine now — this is a rare position for
a manuscript restructure to be in, and the outline below is written to take
advantage of it rather than hedge unnecessarily.

---

## What gets cut, shrunk, or kept (mapped to `MANUSCRIPT_ISSUES.md`)

| Old material | Disposition | Why |
|---|---|---|
| Physics-informed constraints (divergence-free/geostrophic), Issue 1 | **Shrink to one paragraph**, in Methods or Discussion, not a section | The circularity point (GLORYS12V1 is itself NEMO output, so a physics penalty on top of it is predictable-in-advance-inert) is genuinely citable and architecture-agnostic — worth keeping as a stated methodological caution. The full ablation apparatus (Sec. 3.2–3.4/4.4 of the old draft) is not needed to make that point. |
| Patch-vs-whole-domain resolution scaling, Issues 2 & 4 | **Cut entirely** from this paper | DeepONet-specific; never tested on the CNN; Issue 4's discretization-invariance question is now answered more efficiently by citing the literature (below) than by keeping the patch apparatus. Candidate material for a future DeepONet-architecture paper. |
| Weekly-aggregation timescale finding, Issue 6 | **Cut entirely** from this paper | DeepONet-specific, never tested on CNN, and carries its own unresolved forking-paths credibility risk (2-of-6 variables selected post hoc). Not worth the reviewer attention it would draw now that it isn't the headline. |
| CNN vs. DeepONet architecture comparison, Issue 3 | **Keep, condensed** — becomes the paper's motivating comparison, not a full separate study | This is the reason the paper uses a CNN at all; readers need the numbers, not the full multi-attempt saga (CNN-branch DeepONet, DD-DeepONet, patch+CNN-branch — those attempts stay in `RESEARCH_LOG.md`, don't need to appear in the manuscript). |
| Eddy-tracking double-penalty null result, Issue 5 | **Keep, now the paper's Act 2**, updated with the full-scale/correct-LR numbers (item 35), not the original small-sample DeepONet-only result | This is a much stronger version of the original finding — n=1,962 true eddies vs. the original n=220, and it now includes the important nuance that architecture fixes position error but not recall specifically. |
| AMSE loss fix, RESEARCH_LOG items 24–35 | **New — becomes the paper's Act 3 / climax**, the paper's actual novel contribution | Fully confirmed: 10/10 seed-weight combinations significant on both metrics vs. a matched, seed-paired control (item 35). |
| Soft-FSS loss (items 24–25) | **One short paragraph, framed as a design-choice justification, not a separate finding** | It didn't replicate (item 25's sign flip), but *why* it was tried and abandoned in favor of AMSE is a legitimate, honesty-building methodological note — see Discussion outline below. |
| Issues 7 (split robustness), 8 (t-stat overuse), 9 (verification audit), 11 (evidentiary tiers) | **Still apply, cross-cutting** — address in the new draft's Methods/Discussion/Limitations, not separate sections | These were never architecture-specific; they're about how results are described, and the new draft should get them right from the start rather than retrofit them later. |
| Issue 10 (split into multiple papers) | **Resolved by this restructuring itself** | The decision to cut/defer the DeepONet material *is* the scope decision Issue 10 asked for. State this explicitly in a cover note if the paper goes back to the original reviewer. |

---

## Recommended title (pick one, or use as a starting point)

- *"Grid-Point Skill Is Not Eddy Skill: A Training-Objective Fix for the Double-Penalty Problem in CNN-Based Mesoscale Eddy Forecasting"*
- *"Beyond RMSE: Closing the Double-Penalty Gap in Deep-Learning Ocean Eddy Forecasting with a Spectral Training Loss"*
- *"Why Architecture Isn't Enough: Fixing Grid-Skill/Eddy-Skill Dissociation in Agulhas Current Forecasting"*

---

## Detailed outline

### Abstract

Six-beat structure, in order:
1. Grid-point skill metrics dominate ML ocean-forecasting evaluation; whether
   they translate into feature-level (eddy) forecast quality is rarely
   tested directly against an independent object-tracking algorithm.
2. We test this for the Agulhas Current (GLORYS12V1 reanalysis). Initial
   architecture choice (DeepONet, motivated by discretization invariance)
   is empirically and literature-motivated abandoned in favor of a CNN — one
   sentence, with the headline number (CNN mean skill +0.3797±0.0089 vs.
   DeepONet +0.0412±0.0061, paired t=-81.99, 5 seeds).
3. Despite the CNN's order-of-magnitude grid-skill advantage, eddy-tracking
   improvement over persistence is inconsistent: architecture reliably fixes
   eddy position error (5/5 seeds significant) but not detection recall
   (1/5 seeds significant) — the double-penalty problem (Ebert 2008).
4. This gap closes at the training-objective level, not the architecture
   level: an AMSE loss (adapted from Subich et al. 2025's global-weather
   fix, via 2D FFT) improves both position error and recall significantly
   against a matched, seed-paired control (10/10 seed-weight combinations,
   worst-case p=0.016).
5. One clause noting the physics-circularity methodological point as a
   secondary contribution.
6. Explicit statement of novelty: first application of this loss family to
   regional mesoscale eddy forecasting; first validation of an
   object-aware training loss via an independent tracking algorithm rather
   than the loss's own metric family.

### 1. Introduction

- Open with the scientific motivation: Agulhas Current mesoscale eddies,
  western boundary current dynamics, downstream forecasting relevance —
  reuse from the old manuscript's opening, largely unchanged.
- Survey recent ML ocean-forecasting work briefly (WenHai/Cui et al. 2025,
  OceanNet, general neural-operator-for-ocean literature) — condense the old
  Sec. 1's literature review, cut anything specific to DeepONet's
  theoretical motivation beyond one paragraph.
- State the central methodological gap directly: point-to-point metrics
  (RMSE, skill scores, ACC) are known to suffer the double-penalty problem
  in spatial verification (cite Ebert 2008; Gilleland et al. 2009; Roberts
  & Lean 2008 for the neighborhood/FSS remedy). WenHai and OceanNet both
  already flag this and address it *at evaluation time* (neighborhood
  metrics, modified Hausdorff distance); this paper's contribution is a
  fix *at training time*, validated independently.
- One paragraph: why not DeepONet — discretization invariance was the
  original rationale (cite Raonić et al. 2023; "The False Promise of
  Zero-Shot Super-Resolution in Machine-Learned Operators" 2025; "Is
  Zero-Shot Super-Resolution Possible in Operator Learning?" 2026
  impossibility theorem — all establish that DeepONet's namesake advantage
  doesn't hold on the input/branch side). Forward-reference the empirical
  comparison in Results rather than presenting the numbers twice.
- Close with an explicit, numbered statement of contributions (3, matching
  the three-act structure).

### 2. Data & Methods

- 2.1 GLORYS12V1 reanalysis, domain, resolution (r=6), train/val/test split
  — reuse from old manuscript essentially unchanged.
- 2.2 Architecture: CNN/U-Net (`ConvBlock`-based, ~434K params) as the
  primary model. One paragraph on the DeepONet comparison model
  (whole-domain, branch-trunk, dense-MLP branch, 14.2M params) purely as
  the comparison baseline — full architectural detail can be trimmed
  relative to the old draft, since it's no longer the paper's subject.
- 2.3 Training: loss functions, learning rate (flag the established
  CNN-optimal `lr=1e-3` explicitly, since a wrong-LR bug was a real
  mid-project issue — see the honesty note under Discussion), persistence
  residual convention if retained.
- 2.4 AMSE loss: full technical description non-negotiable here since this
  is the paper's core method. Cover: (a) Parseval decomposition of MSE into
  a per-wavenumber spectral-amplitude term and a decorrelation/coherence
  term; (b) the blurring incentive in vanilla MSE (a model can shrink the
  decorrelation penalty by suppressing its own amplitude); (c) the fix —
  replacing the geometric-mean prefactor with `max(PSD_pred, PSD_true)`;
  (d) the domain adaptation — 2D FFT with radial-wavenumber binning in
  place of Subich et al.'s spherical harmonics (state plainly that
  spherical harmonics were a consequence of GraphCast being a global
  lat/lon model, not load-bearing to the mechanism); (e) that it's applied
  as a weighted auxiliary term on top of pointwise MSE (`amse-weight`),
  not as the sole loss the way Subich et al. used it — a deliberate
  departure, framed as consistent with this project's existing ablation
  convention (physics losses, the earlier FSS attempt). Cite the
  Parseval self-test (`_selftest()`) as evidence of the same
  verify-before-trust discipline the paper already applies elsewhere
  (ties into Issue 9).
- 2.5 Eddy detection and tracking: py-eddy-tracker pipeline, recall and
  position-error metrics, state the `pixel_limit=(1,2000)` filter setting
  plainly as a limitation (not the field-standard 4px threshold) — see the
  open-items section below for whether to close this before submission.
- 2.6 Statistical testing: multi-seed convention (5 seeds, matching
  `cnn_seed_sweep.slurm`'s precedent throughout), paired t-tests for
  grid-skill comparisons, paired bootstrap/permutation testing for
  eddy-tracking comparisons — reuse the old manuscript's Sec. 2.7
  largely unchanged, but apply Issue 8's caution (distinguish "not
  distinguishable from seed noise" from "a large, generalizable effect")
  consistently from the start rather than retrofitting it.
- 2.7 (short) Physics-informed constraints: state that soft
  divergence-free/geostrophic penalties were tested in earlier
  development on the DeepONet architecture and found statistically inert;
  note the reanalysis-circularity explanation (GLORYS12V1 is itself NEMO
  output, so a penalty rewarding physics-consistency the labels already
  satisfy has no gradient to supply) as the likely mechanism, framed as a
  stated methodological finding rather than relitigated as a full ablation.

### 3. Results

- 3.1 **Architecture comparison: CNN vs. DeepONet.** The condensed version
  of Issue 3's full evidentiary trail. Table: single-step mean skill
  (+0.3797±0.0089 vs. +0.0412±0.0061, t=-81.99), rollout skill by horizon
  (1d/5d/10d significant, t=-82.27/-35.72/-9.67; 20d converges,
  t=+0.38 — both architectures go negative under enough autoregressive
  compounding). One sentence noting the parameter-count contrast (434K vs.
  14.2M, 33x). This section's job is to justify using a CNN, not to
  re-litigate every DeepONet-branch-fix attempt from `RESEARCH_LOG.md` —
  leave those out.
- 3.2 **Grid-point skill is not eddy skill.** Present the double-penalty
  result at its final, correct-LR, 5-seed, full-scale (n=1,962 true
  eddies) form — this is item 35's control table, not the original
  single-seed/small-sample DeepONet result. Recall vs. persistence: only
  1/5 seeds individually significant (mean +0.0138±0.0082). Position error
  vs. persistence: 5/5 significant (mean −3.891±0.211 km). Frame this
  explicitly as double-penalty (Ebert 2008): architecture-driven grid-skill
  gains translate reliably into positional accuracy but not into reliable
  detection.
- 3.3 **The AMSE fix.** This is the paper's climax section — give it the
  most space and the cleanest tables.
  - vs. persistence (item 34's numbers, still valid): both weights (0.01,
    0.02), 5 seeds each, significant on both metrics, worst-case p<0.0001.
  - vs. the matched, seed-paired CNN control (item 35's numbers — cite
    these, not item 34's mismatched-anchor version): **10 of 10
    seed-weight combinations significant on both metrics**, worst-case
    p=0.016. Aggregated: weight 0.01 recall +0.0402±0.0095, pos. err.
    −1.695±0.265 km; weight 0.02 recall +0.0367±0.0085, pos. err.
    −1.333±0.203 km.
  - Note the two weights are statistically indistinguishable from each
    other on both grid skill and eddy-tracking metrics — no need to force
    a single "winning" weight; report both, or pick 0.01 since it also
    shows a small, real grid-skill edge over the control (item 33: 5-seed
    mean skill +0.3810±0.0071).
- 3.4 **Secondary observation (optional, clearly hedged): rollout
  stability.** Weight 0.01 and 0.02 are the only configurations in this
  entire study with non-negative mean 20-day rollout skill (+0.023±0.074
  and +0.060±0.057 respectively, vs. the plain CNN's −0.094±0.111 and
  DeepONet's −0.077±0.016). Present as an intriguing, unexplained
  correlate, not a mechanistically established finding — the physically
  plausible story (SSH-coupled velocity fields benefiting from AMSE's
  zos-targeted spectral term via the geostrophic relationship) is
  post-hoc and should be labeled as a hypothesis for future work, not a
  claimed mechanism.

### 4. Discussion

- Interpret the mechanism: AMSE's decorrelation term is fundamentally a
  spectral coherence/phase-alignment penalty — directly relevant to
  getting eddy *structure* in the right place, which maps naturally onto
  both position error and (once the confound of comparing every seed
  against a single, mismatched control seed was corrected, item 35) recall
  too.
- **Honesty paragraph, non-negotiable:** the recall/position-error
  relationship was not stable across this investigation's own history —
  an earlier, imperfectly-controlled comparison (item 34) suggested AMSE's
  benefit was concentrated in position error specifically; the corrected,
  properly seed-paired comparison (item 35) shows both metrics improving
  uniformly. State this directly rather than only reporting the final
  numbers — it's a legitimate methodological lesson (comparison design
  matters as much as sample size) and pre-empts a reviewer finding the
  inconsistency in the supplementary data themselves.
- **Second honesty paragraph:** a differentiable neighborhood-based
  (soft-FSS) loss, following the spatially-enhanced-loss-function
  tradition (Lagerquist & Ebert-Uphoff 2022), was tried first and
  abandoned — its initial single-seed result looked promising but flipped
  sign, significantly, across additional seeds (RESEARCH_LOG item 25).
  Frame this as informative: illustrates real seed-sensitivity risk in
  this class of object-aware loss, and motivates why the spectral AMSE
  formulation (with its exact Parseval decomposition, rather than a
  softened/thresholded detection proxy) was the one that ultimately
  replicated cleanly.
- Discuss the recall-null-result reframing at full statistical power: even
  the plain CNN control's recall improvement over persistence, while not
  individually significant at most seeds, is directionally positive at
  every seed and significant at one (2026, p=0.0019) — a substantially
  different picture from the original manuscript's small-sample (n=220)
  "exact tie" framing. State plainly that at least part of the original
  double-penalty "null" result looks like it was partly a statistical-power
  artifact of testing at n~150–220 true eddies, not a hard architectural
  ceiling — this is itself a methodologically useful point for the field,
  connects to Issue 5's original filter-threshold concern.
- Physics-circularity paragraph (Issue 1): state as a citable
  methodological caution — testing physics-informed constraints against
  physics-model-derived reanalysis labels is close to circular by
  construction; note this appears to be an underexplored point in print
  despite an extensive PINN soft-constraint-critique literature (Wang et
  al. 2021; Krishnapriyan et al. 2021) addressing a different mechanism
  (optimization pathology, not label circularity).
- Limitations, stated explicitly and not buried: (1) eddy detection still
  uses the loosened `pixel_limit=(1,2000)` filter, not the field-standard
  4px threshold — note that the full-scale sample (n=1,962) substantially
  addresses the original small-sample concern even under the loose filter,
  but the standard-filter comparison has not been run; (2) single
  chronological train/test split — no cross-period robustness check
  (Issue 7); (3) AMSE's FFT-bin count and threshold/softness
  hyperparameters were chosen once and not swept; (4) results are specific
  to the Agulhas Current / GLORYS12V1 — generalization to other western
  boundary currents or reanalysis products is untested.

### 5. Conclusion

- Restate the three-part contribution in one tight paragraph.
- Future work: standard-filter eddy-tracking rerun, cross-split robustness
  check, extension to other current systems, and — one sentence, no
  detail — note that the shelved DeepONet architecture study (physics
  ablation, patch-resolution scaling, discretization invariance) remains a
  well-documented candidate for separate future work.

### Optional supplementary material

- Full DeepONet architecture-fix attempts (CNN-branch DeepONet, DD-DeepONet,
  patch+CNN-branch) — only if a reviewer specifically asks "did you try to
  fix DeepONet rather than abandon it," otherwise leave in
  `RESEARCH_LOG.md` only.
- Full AMSE weight-sweep grid-skill table (all four weights, both LR
  regimes) — useful supplementary evidence of the LR-correction process
  for a reviewer who wants to see the robustness work, but not necessary
  in the main text.

---

## Open items to resolve before or during drafting (flag to the user, don't decide unilaterally)

1. **Issue 5's standard-filter question.** Given the full-scale sample
   (n=1,962) already substantially outgrows the original small-sample
   concern even under the loosened filter, is the r=3 + standard-4px rerun
   still worth doing before submission, or is this better stated as an
   explicit limitation? This is a real scope/time tradeoff, not something
   to resolve by assumption.
2. **A missing paired test.** `RESEARCH_LOG.md` item 33 reports weight
   0.01's 5-seed grid skill (+0.3810±0.0071) as numerically above the
   official CNN control's 5-seed headline (+0.3797±0.0089) but never runs
   `compare_cnn_vs_deeponet.py`-style paired significance test between
   them specifically (only weight 0.01 vs. weight 0.02 was tested, t=8.73).
   Worth adding — it's a cheap, already-scripted comparison, and "AMSE
   improves grid skill too, significantly" would be a nice bonus claim if
   it holds.
3. **Evidentiary-tier language (Issue 11).** The new draft is in the
   fortunate position of having almost everything at the same high rigor
   tier — worth stating this explicitly in the abstract/conclusion (e.g.,
   "all central comparisons in this paper are confirmed at 5 seeds, full
   scale, with paired significance testing") rather than leaving tier
   language implicit, since it's now a genuine strength rather than a
   patchwork to apologize for.

## Practical notes for whoever executes this

- Both `Agulhas_DeepONet_Manuscript.md` (source of truth) and `.tex` (hand
  mirror) need to change identically — this is a full rewrite of most
  sections, not a patch, so budget for rebuilding the `.tex` from the `.md`
  rather than trying to diff-patch the old LaTeX.
- Rebuild the PDF with `tectonic Agulhas_DeepONet_Manuscript.tex` once both
  source files are updated.
- Log the restructuring itself as a new numbered entry in `RESEARCH_LOG.md`
  once executed, and update `MANUSCRIPT_ISSUES.md`'s statuses per the
  disposition table above (Issues 2, 4, 6 → resolved-by-scope-cut; Issue 10
  → resolved; Issues 1, 3, 5 → status changed to reflect their new role).
- Don't carry over the debugging narrative (the LR bug, the mismatched-seed
  control bug) into the manuscript prose itself beyond the one honesty
  paragraph specified above — that level of process detail belongs in
  `RESEARCH_LOG.md`, not the paper.
