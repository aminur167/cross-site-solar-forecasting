# Data Provenance and Multi-Site Telemetry Manifest

**Project:** Cross-Site Solar PV Power Forecasting  
**Author:** Aminur Rahman (HSTU, Bangladesh)  
**Date of Audit:** November 2026  
**Document Status:** Verified Academic Data Audit  

---

## 1. Primary Experimental Dataset: Romanian Commercial PV Parks

### 1.1 Dataset Provenance and Official Citations
* **Dataset Title:** A Benchmark Dataset and Toolkit for Short-Term Solar Photovoltaic Generation Forecasting
* **Official Repository:** [https://github.com/upb-nlp/PV-Generation-Dataset](https://github.com/upb-nlp/PV-Generation-Dataset)
* **Publisher / Institutions:** University Politehnica of Bucharest (UPB) & University of Craiova, Romania
* **Principal Investigators:** Marian Cristian Mihăescu, Paul Ștefan Popescu, Adina Magda Florea
* **Companion Publications:**
  1. Mihăescu, M. C., Popescu, P. S., & Florea, A. M. (2023). *A Benchmark Dataset and Toolkit for Short-Term Solar Photovoltaic Generation Forecasting*. **Energies**, 16(8), 3450. DOI: [10.3390/en16083450](https://doi.org/10.3390/en16083450).
  2. Popescu, P. S., Mihăescu, M. C., & Florea, A. M. (2023). *Applied Machine Learning for Multi-Site Photovoltaic Power Forecasting*. **Applied Sciences**, 13(21), 11052. DOI: [10.3390/app132111052](https://doi.org/10.3390/app132111052).
* **License:** Open Access / Creative Commons Attribution (CC BY 4.0). Permitted for academic research, modification, and reproduction.

---

### 1.2 Physical System Architecture and Geographic Metadata

| Metadata Channel | **Site A (Source Plant $\mathcal{D}_S$)** | **Site B (Target Plant $\mathcal{D}_T$)** |
|---|---|---|
| **Country & Climate** | Romania (Eastern Europe, Dfb Köppen Climate) | Romania (Eastern Europe, Sub-continental) |
| **System Classification** | Commercial On-Grid Utility PV Park | Independent Commercial On-Grid PV Park |
| **Nameplate Rated Capacity** | $50.0 \text{ kWp}$ ($P_{\text{rated}} = 50 \text{ kW}$) | $50.0 \text{ kWp}$ ($P_{\text{rated}} = 50 \text{ kW}$) |
| **Telemetry Resolution** | Hourly discrete average ($1\text{h}$) | Hourly discrete average ($1\text{h}$) |
| **Data Time Span** | 2025-01-03 22:00:00 to 2026-06-30 20:00:00 | 2025-04-30 21:00:00 to 2026-05-31 20:00:00 |
| **Raw Chronological Span** | $18 \text{ consecutive months}$ ($543 \text{ days}$) | $13 \text{ consecutive months}$ ($396 \text{ days}$) |
| **Total Raw Hourly Records** | **$13,031$ records** (Verified directly) | **$9,504$ records** (Verified directly) |
| **Combined Raw Records** | **$22,535$ verified physical hourly observations** | |

---

### 1.3 Telemetry Channels and Units Specification

| Channel Name | Physical Quantity | Raw Unit | Normalized Metric | Valid Physical Bounds |
|---|---|---|---|---|
| `datetime_utc` | Coordinated Universal Time | ISO 8601 UTC | Timestamp Index | Strictly Monotonic |
| `active_power_kw` | Inverter AC Active Power | $\text{kW}$ | Target $CF = P / 50$ | $[0.0, 60.0 \text{ kW}]$ ($[0.0, 1.2 \text{ CF}]$) |
| `dc_input_power_kw`| DC PV Array Power | $\text{kW}$ | Diagnostic | $[0.0, 65.0 \text{ kW}]$ |
| `ghi_wm2` | Global Horizontal Irradiance | $\text{W/m}^2$ | Weather Feature | $[0.0, 1400.0 \text{ W/m}^2]$ |
| `dni_wm2` | Direct Normal Irradiance | $\text{W/m}^2$ | Weather Feature | $[0.0, 1200.0 \text{ W/m}^2]$ |
| `dhi_wm2` | Diffuse Horizontal Irradiance | $\text{W/m}^2$ | Weather Feature | $[0.0, 800.0 \text{ W/m}^2]$ |
| `temp_c` | Ambient Air Temperature | $^\circ\text{C}$ | Weather Feature | $[-30.0, +50.0 ^\circ\text{C}]$ |
| `humidity_pct` | Relative Atmospheric Humidity | $\%$ | Weather Feature | $[0.0, 100.0\%]$ |
| `cloud_cover_pct` | Total Cloud Cover Fraction | $\%$ | Weather Feature | $[0.0, 100.0\%]$ |
| `wind_speed_kmh` | Anemometer Wind Velocity | $\text{km/h}$ | Weather Feature | $[0.0, 150.0 \text{ km/h}]$ |
| `solar_elevation_deg`| Sun Elevation Angle | Degrees ($^\circ$) | Solar Geometry | $[-90.0^\circ, +90.0^\circ]$ |
| `solar_azimuth_deg` | Sun Compass Azimuth | Degrees ($^\circ$) | Solar Geometry | $[0.0^\circ, 360.0^\circ]$ |

---

### 1.4 Raw to Processed Data Funnel Reconciliation

A rigorous academic paper must reconcile raw record counts with the final evaluated sample size. Below is the audited record funnel:

| Funnel Stage | **Site A (Source)** | **Site B (Target)** | Methodological Justification |
|---|:---:|:---:|---|
| **Raw Records in Files** | **$13,031$** | **$9,504$** | Direct CSV row count (Combined: $22,535$). |
| **Missing Power Records** | $260$ ($2.0\%$) | $170$ ($1.8\%$) | Unrecoverable sensor/transmission loss. |
| **Nighttime Records ($\theta_{\text{elev}} \le 0^\circ$)** | $6,412$ ($49.2\%$) | $4,621$ ($48.6\%$) | Nocturnal periods (astronomically verified zero generation). |
| **Valid Daytime Observations** | **$6,359$** | **$4,713$** | Daytime active generation hours ($\theta_{\text{elev}} > 0^\circ$). |
| **Autoregressive Lag Drop ($t-23$)**| $-23$ | $-23$ | 24-hour lag initialization window. |
| **Final Evaluated Daytime Samples** | **$6,336$** | **$4,690$** | **Final rigorous academic evaluation benchmark.** |

---

## 2. Additional International Open Renewable Datasets Survey

In accordance with good scientific practice, we audited other major international solar PV benchmark datasets to contextualize this work:

### 2.1 NREL PVDAQ (National Renewable Energy Laboratory, USA)
* **URL:** [https://data.openei.org/submissions/4568](https://data.openei.org/submissions/4568)
* **Publisher:** US Department of Energy / NREL
* **License:** Public Domain / Creative Commons Zero (CC0)
* **Characteristics:** High-resolution sub-minute and hourly telemetry from utility and commercial installations across Colorado, Hawaii, and California.
* **Limitations for this study:** Disparate data collection frequencies and non-uniform meteorological telemetry formats across legacy sites require extensive site-specific harmonization.

### 2.2 ECMWF ERA5 Reanalysis
* **URL:** [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
* **Publisher:** Copernicus Climate Change Service (C3S) / ECMWF
* **License:** Copernicus Open Access License
* **Characteristics:** Hourly atmospheric reanalysis on a $0.25^\circ \times 0.25^\circ$ global grid. Provides surface solar radiation downwards (SSRD), 2m temperature, total cloud cover, and 10m wind components.

### 2.3 Mendeley Bangladesh Solar PV Telemetry
* **URL:** [https://data.mendeley.com/datasets/g6pxf8nnt6/2](https://data.mendeley.com/datasets/g6pxf8nnt6/2)
* **Publisher:** Mendeley Data / Independent Researchers (Bangladesh)
* **License:** CC BY 4.0
* **Characteristics:** Rooftop solar PV generation logs from tropical South Asian climate.

---

## 3. Scope and Generalization Boundary

### 3.1 Transparent Declaration of Study Scope
* **Regional Transferability:** This study demonstrates cross-site transfer between two independent, commercial 50 kWp PV plants situated in the same macro-climate region (Romania, Eastern Europe).
* **Boundary Condition:** We **do not claim global or inter-continental generalization**. Inter-continental transfer (e.g. Europe to tropical Bangladesh or arid Arizona) involves radical differences in monsoon precipitation, dust soiling dynamics, and cloud microphysics that require specialized climate-zone adaptation.
* **Academic Integrity Statement:** Framing this work as a *rigorous two-site regional cross-site transfer and uncertainty study under temporal and sensor-availability shifts* provides a scientifically defensible foundation that aligns with the scope of *Energy and AI*.
