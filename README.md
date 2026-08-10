# Grid-Point Skill Is Not Eddy Skill

Code and manuscript sources for **"Grid-Point Skill Is Not Eddy Skill: A Training-Objective
Fix for the Double-Penalty Problem, With an Unexplained Resolution Sensitivity — A Case
Study in the Agulhas Current."**

Read the paper: [`Agulhas_DeepONet_Manuscript.pdf`](Agulhas_DeepONet_Manuscript.pdf)
(supplementary material: [`Agulhas_DeepONet_Supplement.pdf`](Agulhas_DeepONet_Supplement.pdf)).

## What's here

- `Agulhas_DeepONet_Manuscript.{md,tex,pdf}`, `Agulhas_DeepONet_Supplement.{md,tex,pdf}` —
  manuscript sources. The `.md` is the source of truth; the `.tex` is a hand-mirrored twin
  used to build the PDF.
- `manuscript_figures/` — the figures referenced in the manuscript.
- `train_*.py` — training scripts for every architecture/loss variant discussed in the
  paper (CNN baseline, CNN+AMSE, whole-domain and patch DeepONet variants, etc.).
- `*.slurm` — the exact SLURM job scripts used to run training at full scale on the Yale
  YCRC "Bouchet" cluster.
- `eddy_tracking/` — the eddy-detection/tracking pipeline (wraps py-eddy-tracker),
  statistical tests, and the analysis scripts behind every eddy-tracking table in the paper.
- `aggregate_seed_sweep.py`, `compare_cnn_vs_deeponet.py`, `extract_rollout_rmse.py`, and
  other root-level scripts — result aggregation and evaluation utilities.
- `sparse_reconstruction_*.py`, `sparse_track_query_test.py` — a related, separate line of
  work (sparse observational reconstruction), not part of this paper's central result.
- `RESEARCH_LOG.md` — chronological, numbered log of the project's development, including
  four disclosed methodological corrections (see manuscript Sec. 4.2 / Supplement S1).
- `MANUSCRIPT_ISSUES.md` — issue-by-issue record of external review and how each point was
  addressed.
- `docs/history/` — earlier planning/handoff documents, kept for provenance; not current
  project status (see `RESEARCH_LOG.md` for that).

## Reproducing results

1. Data: [GLORYS12V1](https://doi.org/10.48670/moi-00021) (Copernicus Marine Service), a
   public reanalysis product. `download_agulhas_prototype.py` fetches the Agulhas-domain
   subset used here; `prepare_weekly_cache.py` / `build_cache.slurm` build the training
   cache from it.
2. Train: pick a `train_*.py` script and its matching `*.slurm` job file.
3. Evaluate: `aggregate_seed_sweep.py`, `extract_rollout_rmse.py`, and the scripts in
   `eddy_tracking/` reproduce this paper's tables from a run's output directory.

Trained model checkpoints, prediction arrays, and full result directories are not stored
in this repository (they run to tens of GB). Per the manuscript's Code and Data
Availability statement, these are intended to be made available separately.

## License

MIT — see [`LICENSE`](LICENSE).
