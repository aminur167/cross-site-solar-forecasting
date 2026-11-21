# Final Test and Reproducibility Report

## Commands and outcome

| Command | Outcome |
|---|---|
| `python src/data/prepare_datasets.py` | Completed; refreshed processed Site A/B features. |
| `python run_full_pipeline.py --skip-dl` | Completed successfully after fixing a plotter/schema mismatch. Includes data audit, baseline, transfer, CQR/stress, SHAP, Figures 1–6 (Figure 5 when enabled), LaTeX tables, and tests. |
| `python experiments/run_deep_learning_benchmark.py` | Completed separately with corrected target alignment and gap-aware sequence windows. |
| `python -m unittest tests/test_pipeline.py -v` | 18 tests passed. |
| `python -m py_compile ...` | Passed after final code edits: 18 changed Python modules/scripts compile. |
| LaTeX compile | Blocked: `pdflatex` and `latexmk` unavailable in this environment. |

## Environment observed

Python 3.14; NumPy 2.4.6; pandas 3.0.3; SciPy 1.18.0; scikit-learn 1.9.0; LightGBM 4.7.0; XGBoost 3.4.1; PyTorch 2.15.0; SHAP 0.54.1. `pvlib` was unavailable; experiments used the solar geometry supplied in the source dataset. `requirements.txt` is not fully version-pinned.

## Rerun scope

Processed rows: Site A 12,675, Site B 9,259. Holdout counts vary by experiment; the cross-site holdout had 966 target-daytime samples, and source training had 8,605 rows. Final serialized JSON files are under `experiments/results/`. Results are retrospective and the weather inputs were unavailable at forecast issue time.

The first integrated run stopped at Figure 3 because the visualization expected flat metrics while the new five-seed output was nested. The plotter was fixed, remaining outputs regenerated, and `--skip-dl` pipeline rerun successfully; deep learning was run separately.

## Git and package

Creating `research-integrity-complete-repair` was attempted but denied because `.git` metadata is read-only. No commit or push occurred. `paper_submission_package.zip` was rebuilt as a research-review package, not a submission-ready or compiled package.
