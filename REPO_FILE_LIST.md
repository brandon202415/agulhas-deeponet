# GitHub repository file manifest

Written 2026-08-08 as an inventory of what should (and should not) go into the public
GitHub repository for this project. Based on directly walking the current working
directory — not a general template — so paths and sizes below are real.

The current `.gitignore` already excludes `data/`, `results/`, `.venv/`, `__pycache__/`,
`*.pyc`, `.DS_Store`, and `logs/`. That's a good start but doesn't cover everything large
or stale that's actually sitting in the repo (see "Also exclude" below) — the biggest gap
is `eddy_tracking/`, which is 23 GB and isn't gitignored at all right now.

## Include — manuscript sources (the actual deliverable)

- `Agulhas_DeepONet_Manuscript.md` — source of truth
- `Agulhas_DeepONet_Manuscript.tex` — hand-mirrored LaTeX twin
- `Agulhas_DeepONet_Manuscript.pdf` — built output (fine to commit; it's small, ~870 KB, and lets the repo double as "read the paper here")
- `Agulhas_DeepONet_Supplement.md` / `.tex` / `.pdf` — same pattern, supplementary material
- `manuscript_figures/fig_domain_map.png`
- `manuscript_figures/fig_amse_schematic.png`
- `manuscript_figures/fig_forest_plot.png`
- `manuscript_figures/fig_eddy_qualitative.png`
  (the 4 figures actually referenced by the current manuscript — see "Exclude" for the 7 stale ones sitting alongside them)

## Include — code that produces the manuscript's results

**Training scripts** (all `train_*.py` at repo root — 15 files): these define every
architecture/variant referenced anywhere in the manuscript or research log (CNN baseline,
CNN+AMSE, CNN+FSS-loss, whole-domain DeepONet, patch DeepONet and its attention/dd/cnnbranch
variants, continuous-lead-time variants, resolution-augmentation, satellite-holdout). Include
all of them — even the ones behind secondary/superseded findings, since `RESEARCH_LOG.md`
and the Supplement reference specific ones by name and a reader trying to verify a claim
needs the actual script, not just its filename in a log.

**Analysis / evaluation scripts** (repo root):
`aggregate_seed_sweep.py`, `compare_cnn_vs_deeponet.py`, `discretization_invariance_test.py`,
`evaluate_weekly_climatology_baseline.py`, `extract_rollout_rmse.py`,
`test_lr_config_regression.py`, `test_physics_losses_synthetic.py`,
`validate_satellite_fullscale.py`, `download_agulhas_prototype.py`, `prepare_weekly_cache.py`

**Figure generation** (repo root): `make_figures.py` (older, pre-restructuring figure set —
keep for provenance even though its outputs are now stale, see below),
`make_fig_domain_map.py`, `make_fig_amse_schematic.py`, `make_fig_forest_plot.py`
(these three are new this session and produce the 4 current manuscript figures, together
with the eddy_tracking one below)

**Sparse-reconstruction exploration** (repo root, a distinct research thread mentioned in
`RESEARCH_LOG.md` as future-paper material): `sparse_reconstruction_attention.py`,
`sparse_reconstruction_multifidelity.py`, `sparse_reconstruction_prototype.py`,
`sparse_reconstruction_setenc.py`, `sparse_track_query_test.py` — include for completeness;
they're real code backing a documented line of work, not clutter.

**`eddy_tracking/` code** (9 `.py` files at its root):
`eddy_tracking_analysis.py` (the core module — `identify()`, `match_with_detail()`, etc.,
reused unmodified by every other script here), `eddy_stat_test.py`,
`domain_review_analysis.py`, `eddy_uv_confound_check.py`, `find_qualitative_example.py`,
`make_fig_eddy_qualitative.py`, `test_synthetic_ground_truth.py`,
`test_control_pairing_regression.py`, `sanity_check_plot.py`. Also `eddy_tracking/README.md`, plus `eddy_tracking/aviso/aviso_true_obs.py` and
`eddy_tracking/aviso/fetch_essential_fields.py` — the two small loader scripts in that
folder, kept separate from the two giant `.nc` files sitting next to them (see Exclude).

## Include — SLURM job scripts (all 27, repo root)

Every `*.slurm` file. These are small text files (2–8 KB each) and are exactly what a
reader needs to actually reproduce a cluster run — excluding them would defeat the point
of the code/data availability statement now in the manuscript's Conclusion.

## Include — project documentation

- `CLAUDE.md` — operational conventions; genuinely useful to anyone (human or agent) picking this repo up
- `RESEARCH_LOG.md` — the closest thing this project has to a commit history (per `CLAUDE.md` itself); load-bearing, keep
- `MANUSCRIPT_ISSUES.md` — the external-review issue tracker; also load-bearing, directly explains why the manuscript reads the way it does
- `ONBOARDING.md` — still useful as a project-status summary, though it's showing its age (last substantive update 2026-07-30) and could use a refresh before publishing

## Judgment calls — documentation

These four are real project artifacts, not junk, but they're written *for a future AI
agent session* or capture a *since-resolved* decision point rather than being written for
a human visitor to a public repo:

- `HANDOFF.md`, `HANDOFF_CONTEXT.md` — session handoff notes ("written for a fresh agent
  picking up this session"); accurate history, but the voice/audience is wrong for a public
  README-adjacent file
- `MANUSCRIPT_RESTRUCTURE_PLAN.md` — explicitly a superseded planning doc ("Status: Decision
  made 2026-07-30, not yet executed" — it has since been executed)
- `TECHNICAL_WALKTHROUGH.md`, `RESULTS.md` — both look like they predate the current
  manuscript's restructuring and may now disagree with it in places (worth a diff-check
  against the current manuscript before including, not a blind include)

**Recommendation:** move these four into a `docs/history/` or `notes/` subfolder rather
than dropping them at repo root, or fold anything still-accurate into `ONBOARDING.md` and
drop the rest. Don't delete them outright — they're real project history — but don't let
a visitor mistake them for current documentation either.

## Include — small, essential result/analysis artifacts

- `eddy_tracking/uv_confound_results/` (8.8 MB, 5 JSON files) — this session's uo/vo
  confound check, directly backs the Limitations "Eighth" point's numbers
- `eddy_tracking/sanity_check/`, `eddy_tracking/domain_review_r3_mplfix/`,
  `eddy_tracking/domain_review_r6_mplfix/` (all under 100 KB combined) — small, cheap to
  keep, referenced in the research log
- The 140 root-level `eddy_tracking/*.json` files (39 MB total, `detailed_*.json` and
  `eddy_tracking_results*.json`) are a judgment call, not a clean include — see below

## Judgment call — the 140 root-level eddy_tracking JSON files

39 MB total, individually small (avg ~280 KB), so size isn't the blocker — GitHub is fine
with this. The problem is **provenance**: this session already found and flagged one of
these exact files (`detailed_cnn_seed2026.json`) as silently holding pre-matplotlib-fix,
232×-undercounted eddy counts from *before* this project's central bug fix — i.e., some
fraction of these 140 files are known-stale results from superseded pipeline states, sitting
at the same path depth as the current, correct ones, with nothing in the filename
distinguishing them. Before committing this directory wholesale:

1. Cross-reference each file's `summary.n_true_eddies_total` (or equivalent) against the
   manuscript's stated counts (n=2,336,570 pooled at r=3, n=454,662 per-seed at r=6) to sort
   current from stale.
2. Either delete the stale ones outright, or move them to a clearly-labeled
   `archive/superseded/` path with a one-line note of what was wrong with each.
3. Only then commit the survivors.

Committing this directory without that pass risks exactly the failure mode this project's
own `RESEARCH_LOG.md` and `MANUSCRIPT_ISSUES.md` spent multiple rounds catching and
disclosing — a stale result file sitting where a reader would reasonably assume it's current.

## Exclude — raw data and large caches (already gitignored)

- `data/` (2.9 GB: `agulhas_prototype.nc`, 4 `cache_*_local.npz` files) — the raw prototype
  netCDF and local-dev subsample caches. GLORYS12V1 is a public Copernicus product;
  `download_agulhas_prototype.py` + `prepare_weekly_cache.py`/`build_cache.slurm`
  regenerate these from the public source. Already correctly gitignored.
- `results/` (tens of GB: every `model.pt`/`predictions.npz`/`metrics.json` from every
  training run) — already correctly gitignored. This is also exactly the material the
  manuscript's new "Code and data availability" paragraph promises to make available
  separately (i.e., not via git — this belongs on something like Zenodo/OSF/institutional
  storage with a DOI, referenced from the repo's README, not committed to git history).

## Exclude — not yet gitignored but should be

- **`eddy_tracking/aviso/`** (11 GB: two ~5.5 GB AVISO altimetry `.nc` files,
  `META4_DT_allsat_{cyclonic,anticyclonic}_19930101_20230908.nc`) — third-party satellite
  data, publicly available from AVISO/CMEMS, far too large for git regardless. **Add
  `eddy_tracking/aviso/*.nc` to `.gitignore`** (keep any small loader script in that folder).
- **`eddy_tracking/results_r3_std_mplfix_check/`** (4.5 GB), **`eddy_tracking/results_r6_mplfix/`**
  (4.4 GB), **`eddy_tracking/results_r3_aviso/`** (2.9 GB) — per-day detailed JSON dumps,
  ~309 MB *per file*, far over GitHub's 100 MB hard limit even one at a time. These are
  intermediate caches regenerable from `results/*/predictions.npz` via
  `eddy_tracking_analysis.py`. **Add these three directories to `.gitignore`.**
- **`eddy_tracking/results_r3_std/`** (23 MB) — this is the *specific stale duplicate*
  `RESEARCH_LOG.md` documents as a pre-matplotlib-fix trap, superseded by
  `results_r3_std_mplfix_check/`. Small enough to not matter for repo size, but it should
  not be committed as if it were valid data. Delete it, or if kept for provenance, move to
  an clearly-labeled archive path — don't leave it sitting at a name that looks current.
- **`manuscript_figures/fig_acc_vs_lead.png`, `fig_eddy_map_day18.png`, `fig_eddy_map_day45.png`,
  `fig_eddy_patch_vs_wholedomain.png`, `fig_rollout_acc_byvar.png`, `fig_skill_singlestep.png`,
  `fig_sweep_skill.png`** (7 files, ~600 KB total) — all dated 2026-07-26/27, all predate the
  AMSE/CNN restructuring, and none are referenced by `\includegraphics` anywhere in the
  current `.tex`. Either delete or move to an archive folder; committing them next to the
  4 actually-used figures with no distinction invites exactly the same "which one is current"
  confusion as the stale JSON files above.

## Exclude — environment / OS / tool artifacts (already gitignored, confirmed still correct)

- `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store` — already in `.gitignore`, correctly so
  (macOS `.venv` won't run on the cluster's Linux environment; `__pycache__` patterns
  without a leading slash already match `eddy_tracking/__pycache__/` too, confirmed)
- `.claude/settings.local.json` — Claude Code's local tool-permission config; not
  gitignored currently but should be — it's machine/session-local, not project content.
  **Add `.claude/` to `.gitignore`.**
- `eddy_tracking/.DS_Store` — covered by the existing `.DS_Store` pattern already.

## Missing, worth adding before making the repo public

- **`README.md`** — there isn't one at repo root right now. Worth a short one: what this
  project is, pointer to the manuscript PDF, how to reproduce (which script → which SLURM
  job → which table), and a link to wherever `results/`/`data/` end up hosted externally.
- **`LICENSE`** — none present. Worth deciding before the repo goes public.
- **`requirements.txt` or `environment.yml`** — none present; the manuscript's new "Code
  and data availability" paragraph implicitly promises reproducibility, and right now
  there's no single file listing the actual package versions (PyTorch, py-eddy-tracker,
  netCDF4, matplotlib, etc.) needed to run any of this. Worth generating from the
  `agulhas`/`eddytrack` conda environments `CLAUDE.md` references.

## Bottom line

A clean "include" set — manuscript sources, all code, all SLURM scripts, docs, and the
curated small result artifacts — comes to roughly 50 MB, comfortably within normal GitHub
limits with no individual file anywhere near the 100 MB hard cap. The three real risks are
(1) `eddy_tracking/aviso/` and the three `*_mplfix*`/`*_aviso` result-dump directories
(23 GB combined, currently *not* gitignored) getting committed by accident, (2) the 140
root-level eddy_tracking JSONs needing a stale-vs-current pass before they're trustworthy
to publish, and (3) the 7 stale `manuscript_figures/` PNGs and 4 internal handoff-style docs
creating exactly the kind of "which version is real" ambiguity this project's own tracking
documents have flagged and fixed multiple times already this session — worth applying the
same discipline here before anything goes public.
