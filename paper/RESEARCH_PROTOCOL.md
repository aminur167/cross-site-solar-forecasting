# Experimental protocol

## Data and target

The study uses hourly records from two roughly 50 kW PV sites in southwestern Romania. Weather fields are Open-Meteo historical estimates, not measurements available when the forecast would have been issued. The work is therefore retrospective.

Each power value is an average over the UTC hour beginning at `datetime_utc`. Row *t* is paired with the following hour’s capacity factor. Daytime means target solar elevation above 5 degrees. Raw twilight and over-nameplate readings are retained with quality flags; the model target is bounded separately.

## Evaluation

All splits are chronological. Site B’s final 20% is the holdout. Source labels at or after the first target test time are excluded; source training uses the first 85% of the remaining rows. Adaptation budgets are right-open calendar windows. Exact counts are in the experiment JSON.

For intervals, the report gives observed PICP, width, and Winkler score. Source calibration uses earlier Site A rows; target calibration uses the first 14 calendar days of Site B’s adaptation pool. Temporal dependence and site shift mean no coverage guarantee is claimed.

Weather-feature corruption tests are synthetic and do not reproduce field sensor faults. The LSTM and TCN were each run once. Their DM comparison uses the longest contiguous all-hours segment, so those results are exploratory.
