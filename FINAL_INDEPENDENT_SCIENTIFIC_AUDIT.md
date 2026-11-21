# Final Independent Scientific Audit

## Outcome

**Scientific status: corrected research draft, not submission-ready.** The two-site experiments ran and outputs were regenerated, but important scope and reproducibility limitations remain.

## Findings addressed

| Severity | Finding | Resolution |
|---|---|---|
| Critical | Target-day mask was not aligned to the next-hour label; temporal gaps could be treated as adjacent. | Feature builder now aligns target timestamp/daytime and rejects non-adjacent targets; sequence windows also require hourly continuity. |
| Critical | Raw target power/twilight values were silently altered or night output forced to zero. | Preserve source power and quality flags; analytical target is separately bounded; daytime threshold follows source README (>5 degrees). |
| Critical | Weather variables described as on-site sensors/issue-time inputs although they are retrospective Open-Meteo estimates. | Provenance, manuscript, README and app now state retrospective status; operational claim removed. |
| Critical | Same-region sites represented as distinct microclimates/global transfer. | Scope corrected to two sites in the same southwestern Romanian region. |
| Critical | Prior model metrics, coverage claims and manuscript conclusions were stale or unsupported. | Reprocessed data, reran experiments, regenerated tables/figures and corrected narrative. |
| High | CQR clipping changed interval validity; coverage guarantee overstated under time/site dependence. | Removed clipping; report empirical coverages only and no guarantee. |
| High | Sensor perturbations were presented as real sensor failure/robustness. | Synthetic weather-feature perturbations explicitly labeled; synchronized loss/stuck scenarios added. |
| High | Gap-spanning neural sequences and significance claims were not defensible. | Sequence windows are gap-aware; corrected single-seed results labeled exploratory; DM scope disclosed. |
| High | Known citation metadata was inaccurate/unsubstantiated. | Removed incorrect PV attribution to DOI 10.3390/en16083450, corrected Yang review, removed unverified references; bibliography narrowed. |
| Medium | Old review, author and journal documents contained fabricated PASS/novelty/APC assertions. | Replaced with audit checklist and author verification requirements. |

## Current empirical summary

- Source-only transfer to Site B: 6.93% daytime nMAE; persistence 14.14%.
- One-day pooled adaptation: 6.97%, slightly worse than zero-shot; one-day target-only: 28.06%.
- Nominal 90% daytime PICP: raw 88.20%, source-calibrated 90.27%, target-calibrated 94.51%. Empirical only.
- Five-seed synthetic 30% feature-wise MCAR: mean nMAE 7.46%, mean PICP 92.78%. Not field fault evidence.
- Single-seed deep models: LightGBM 6.93%, LSTM 7.37%, TCN 7.69% daytime nMAE. No broad superiority inference.

## Remaining blockers

1. Input weather is hindsight data, so valid issue-time forecasting is not demonstrated.
2. Only two same-region sites are available.
3. Neural models are single-seed and statistical comparisons use a contiguous all-hours subset.
4. `requirements.txt` is not fully pinned; LaTeX compiler unavailable; generated package is a review draft.
5. Full literature/novelty review and all author statements require human verification.
6. Requested branch could not be created: `.git` is read-only in this environment.

See `RESEARCHER_ACTION_REQUIRED.md` for required author decisions.
