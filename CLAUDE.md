# Agulhas DeepONet — project instructions

**Before doing anything else, read `ONBOARDING.md`.** It has the current state,
the pending decision, and pointers into the two living tracking documents
(`MANUSCRIPT_ISSUES.md`, `RESEARCH_LOG.md`). Do not start executing on a request
in this repo without reading it first — this project has a long history and a
consequential open decision, and skipping the briefing will waste the user's
time re-explaining context that's already written down.

## Operational conventions (apply to every session in this repo)

- **Cluster access.** All full-scale experiments run on the Yale YCRC "Bouchet"
  cluster via SLURM, from `~/scratch_pi_<pi_netid>/eddy`. Standard activation
  sequence in every `.slurm` script: `module load miniconda; source
  "$(conda info --base)/etc/profile.d/conda.sh"; conda activate agulhas`. GPU
  jobs use `--partition=gpu --gpus=rtx_5000_ada:1`; CPU-only prep (e.g. cache
  building) uses `--partition=day` to avoid idle-GPU watchdog kills. You cannot
  run `sbatch` yourself — give the user the exact command and wait for them to
  paste back real output. **Never fabricate, estimate, or predict cluster or
  background-process results.** If asked "is there progress," check the actual
  output file/process state and report honestly, including "not visible yet."

- **Local background processes die at session/context boundaries.** This has
  happened repeatedly. For anything that must survive and produce a trustworthy
  result, either (a) run it via SLURM on the cluster, or (b) run it in the
  foreground with `python3 -u` (unbuffered) so partial progress is visible
  before it's ever at risk of being killed. Piping a long local run through
  `| tail` without `-u` hides all progress until the process exits — don't do
  that for anything you might need to check on mid-run.

- **Local Apple Silicon (MPS) backend.** Some models in this codebase (the CNN
  encoder / GroupNorm+AdaptiveAvgPool2d combination) stall badly on MPS. The
  affected scripts (`train_cnn_baseline.py`, `train_agulhas_deeponet_cnnbranch.py`,
  `train_agulhas_deeponet_patch_cnnbranch.py`) already force CPU locally for
  this reason — the real cluster run always uses CUDA regardless, so this only
  affects local dev convenience. If you add a new local-testable script with a
  conv encoder, consider skipping the MPS branch preemptively.

- **The manuscript.** `Agulhas_DeepONet_Manuscript.md` is the source of truth;
  `.tex` is a hand-mirrored twin (same content, LaTeX-escaped) used to build
  `.pdf`. Any manuscript edit must be made in **both** files identically, then
  rebuilt with `tectonic Agulhas_DeepONet_Manuscript.tex`. Do not edit the
  `.tex` only, and do not edit the `.pdf` directly (it's generated).

- **Git.** This is not a git repository. There is no version history beyond
  what's written into `RESEARCH_LOG.md` — treat that file as the closest thing
  to a commit log, and keep it updated when you do consequential work.

- **Memory discipline.** This project has two living tracking documents that
  must be kept current, not just written once: `MANUSCRIPT_ISSUES.md` (the
  external-review issue tracker, one entry per issue, status updated as work
  lands) and `RESEARCH_LOG.md` (chronological, numbered entries — append, don't
  rewrite history). When you complete a real piece of work — an experiment, a
  confirmed result, a decision — log it in the relevant document(s) before
  moving on. Don't let context that only exists in a chat transcript be the
  only record of it.
