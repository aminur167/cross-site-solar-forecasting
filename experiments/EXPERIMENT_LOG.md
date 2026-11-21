# Scientific Experiment Execution Log

This log summarizes the corrected rerun. Detailed values are serialized in `experiments/results/`; results are retrospective and do not establish operational forecast skill.

## Protocol and environment

- Data rows represent hourly AC power and retrospective Open-Meteo historical weather estimates; source and target are in the same southwestern Romanian region.
- Daytime is target aligned using solar elevation >5 degrees. Raw twilight/over-nameplate values are flagged and preserved.
- Common source cutoff precedes target holdout. Target adaptation/calibration windows are chronological and right-open.
- Current environment: Python 3.14; NumPy 2.4.6; pandas 3.0.3; SciPy 1.18.0; scikit-learn 1.9.0; LightGBM 4.7.0; XGBoost 3.4.1; PyTorch 2.15.0; SHAP 0.54.1. `pvlib` unavailable; supplied source geometry used.
- Dependencies are not fully pinned. See `FINAL_TEST_AND_REPRODUCIBILITY_REPORT.md`.

## Corrected results

- Site A within-site holdout: Gradient Boosting nMAE 6.11%, nRMSE 10.29%, R2 0.9004; see JSON for sample count.
- Site B zero-shot: 6.93% nMAE; persistence 14.14%. One-day pooled adaptation 6.97%; target-only 28.06%. Pooled adaptation did not improve on zero-shot.
- Nominal 90% interval daytime PICP: raw 88.20%, source-calibrated 90.27%, target-calibrated 94.51%. Empirical values only; no exchangeability guarantee under dependence/site shift.
- Synthetic feature-wise 30% MCAR: 7.46% mean nMAE over five perturbation seeds; not field sensor evidence.
- Deep learning single-seed: LightGBM 6.93%, LSTM 7.37%, TCN 7.69% daytime nMAE. DM comparisons use longest contiguous all-hours segment; exploratory only.
- SHAP top feature: lag_0 (54.54% relative attribution); attribution is not causal.

## Execution record

- `python run_full_pipeline.py --skip-dl` completed successfully after a plotting-schema fix; deep-learning benchmark ran separately after it.
- One earlier full run stopped at Figure 3 due a mismatch between aggregated stress JSON and plotting code; the plotter was corrected and the pipeline rerun successfully.
- `python -m unittest tests/test_pipeline.py -v`: 18 tests passed.
- LaTeX compilation was not possible because neither `pdflatex` nor `latexmk` was installed.
- Branch creation was denied by read-only `.git` metadata; changes remain on the current workspace branch. No commit/push/submission was made.
