# Evidence-Bound Manuscript Readiness Checklist

Status: research draft; not submission ready. This checklist records verifiable blockers and does not simulate reviewer approval.

| Item | Status | Evidence / remaining action |
|---|---|---|
| Data source and timing | Documented | Historical Open-Meteo data is not issue-time forecast input; operational validity remains untested. |
| Scope | Corrected | Two sites in one southwestern Romanian region; no global or cross-climate claim. |
| Leakage and target alignment | Code-reviewed and rerun | Target timestamps/daytime mask and contiguous next-hour rows are enforced in preprocessing. |
| Raw data quality | Audited | Twilight/over-nameplate values are flagged and preserved; author must review raw anomalies. |
| Statistical inference | Limited | Neural comparison is single-seed; DM tests use longest contiguous all-hours segment; interpretation is exploratory. |
| Prediction intervals | Limited | Empirical coverage only; temporal dependence/site shift invalidate unqualified exchangeability claims. |
| Citations | Partial | Known DOI/title mismatches removed; full systematic literature and novelty review remains. |
| Reproducibility | Partial | Results rerun in current environment; requirements are not pinned; LaTeX compiler unavailable. |
| Authorship and declarations | Author confirmation required | Verify affiliation, contribution, conflicts, funding, data/code availability and permission to publish. |
| Submission package | Blocked | Rebuild draft ZIP only; compile and independently review before any submission. |

Do not copy proposed reviewer responses or unverified journal/APC statements from earlier project drafts.
