# Structural Breaks and Changepoint Analysis

A computational econometrics project investigating structural changes in regression relationships through Monte Carlo simulation, robustness experiments, and an application to Indian Consumer Price Index (CPI) data.

## Project Overview

Structural breaks occur when the statistical properties or parameters of a time series change over time. If these changes are ignored, a model estimated over the full sample may provide misleading estimates and inference.

This project studies methods for identifying structural changes in regression models. It combines a simulation study with an empirical application to monthly Indian CPI data.

## Research Objectives

- Examine how structural breaks affect regression relationships.
- Evaluate break-detection performance under different simulated data-generating processes.
- Investigate how sample size, error variance, break magnitude, and break location affect detection.
- Apply structural-break procedures to Indian CPI growth data.
- Discuss the limitations of break detection when residual serial correlation is present.

## Methodological Framework

### 1. Chow-style structural-break test

A pre-specified break date is examined using a joint Wald test with heteroskedasticity- and autocorrelation-consistent (HAC) covariance estimation.

In the empirical application, April 2020 is used as the pre-specified date. This is a HAC-robust Wald implementation, rather than the textbook classical Chow F-test.

### 2. Dynamic-programming segmentation with BIC

An exhaustive dynamic-programming segmentation procedure is used to identify break locations by minimizing the residual sum of squares across candidate segmentations. The Bayesian Information Criterion (BIC) is used to select the preferred number of breaks.

The implementation is described as **Bai–Perron-style dynamic-programming segmentation with BIC**. It should not be interpreted as a complete implementation of every element of the formal Bai–Perron testing framework.

### 3. Supplementary procedures

The repository also contains supplementary implementations or exploratory scripts for:

- CUSUM-based stability diagnostics
- Quandt–Andrews-type unknown-break testing
- PELT segmentation

These are supplementary and are not included as primary methods in the reported Monte Carlo performance comparison.

## Part I — Monte Carlo Simulation

The simulation study evaluates five data-generating scenarios. Each scenario is evaluated over 1,000 replications, for a total of 5,000 replications.

| Scenario | Data-generating feature | Specification |
|---|---|---|
| S1 — No break | Stable regression | Intercept = 2, slope = 0.5, error SD = 1 |
| S2 — Intercept break | One intercept change | Break at observation 150; intercept changes from 2 to 5 |
| S3 — Slope break | One slope change | Break at observation 150; slope changes from 0.5 to 1.2 |
| S4 — Multiple breaks | Two coefficient changes | Breaks at observations 100 and 200; intercepts = 2, 5, 3; slopes = 0.5, 1.2, 0.3 |
| S5 — Variance break | Error variance changes | Break at observation 150; error SD changes from 0.5 to 2, with no regression-coefficient break |

The simulation uses a sample size of 300 and a minimum segment length of 30. The maximum candidate number of breaks is constrained by the minimum segment-length requirement.

### Interpretation of the variance-break scenario

The regression segmentation procedure is designed to identify changes in regression coefficients. Scenario S5 changes the error variance but does not change the regression coefficients. Its no-break classification rate therefore measures how often the procedure avoids falsely identifying a coefficient break in this setting. It does **not** measure the procedure's ability to detect a variance break.

## Part II — Robustness Analysis

The robustness experiment contains 14 configurations with 200 replications per configuration, giving 2,800 replications in total.

The experiments vary:

- **Sample size:** 100, 300, and 500
- **Error standard deviation:** 0.5, 1, 2, and 3
- **Intercept-break magnitude:** 0.5, 1, 2, and 4
- **Break location:** 25%, 50%, and 75% of the sample

The robustness data-generating process focuses on a single intercept break. These experiments should not be interpreted as a robustness assessment of slope breaks or multiple-break settings.

## Part III — Empirical Application: Indian CPI

The empirical application uses monthly All-India, Combined, General Consumer Price Index data from January 2013 to December 2024.

| Dataset characteristic | Description |
|---|---|
| Measure | Consumer Price Index (CPI), General |
| Coverage | All India, Combined |
| Frequency | Monthly |
| Period | January 2013 – December 2024 |
| Observations | 144 |

The analysis includes:

1. Data preparation and construction of the CPI time series.
2. Log transformation and unit-root/stationarity diagnostics using ADF and KPSS tests.
3. First differencing to examine monthly CPI growth.
4. Autocorrelation diagnostics for the growth series and model residuals.
5. A HAC-robust joint Wald test at the pre-specified April 2020 date.
6. Dynamic-programming segmentation with BIC, including minimum-segment-length sensitivity checks.

### Main empirical findings

- The log CPI series is consistent with non-stationary behaviour, while its first difference is plausibly stationary under the diagnostics conducted.
- The HAC-robust test at April 2020 produces a p-value of approximately **0.0549**. It is not statistically significant at the 5% level.
- The BIC-selected dynamic-programming segmentation selects **zero breaks** in the monthly CPI growth regression under the specifications examined.
- The zero-break result remains unchanged in the reported minimum-segment-length sensitivity checks (24, 30, and 36 observations).
- Residual autocorrelation remains an important diagnostic concern, including in the alternative autoregressive specifications examined.

The empirical results therefore provide insufficient evidence of structural breaks in the modelled monthly CPI growth relationship under the tested specifications. This conclusion is specific to the data, model, break definition, and procedures used; it does not establish that the CPI process experienced no structural changes of any kind.

## Repository Structure

```text
STRUCTURAL-BREAKS/
├── data/
│   ├── raw/
│   │   └── cpi_1133.xlsx
│   └── processed/
│       └── india_cpi_cleaned.csv
├── notebooks/
│   └── real_data_application.ipynb
├── outputs/
│   ├── figures/
│   │   ├── real_world/
│   │   └── simulation/
│   └── tables/
├── results/
│   ├── monte_carlo_raw.csv
│   └── robustness_raw.csv
├── scripts/
│   ├── monte_carlo.py
│   ├── robustness_analysis.py
│   ├── chow_test.py
│   ├── bai_perron_dp.py
│   ├── cusum_test.py
│   ├── quandt_andrews.py
│   └── pelt_segmentation.py
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The tree above describes the intended project organization. If filenames in your local checkout differ, use the actual repository contents as the definitive structure.*

## Installation and Setup

### 1. Clone the repository

Replace the placeholder with the SSH or HTTPS URL of your GitHub repository.

```bash
git clone <YOUR-REPOSITORY-URL>
cd STRUCTURAL-BREAKS
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `notebooks/real_data_application.ipynb` to explore the empirical application.

Simulation and analysis scripts can be run from the project root. For example:

```bash
python scripts/monte_carlo.py
python scripts/robustness_analysis.py
```

Check the script-level configuration and output paths before running a full experiment, particularly if you want to reproduce the original replication counts.

## Dependencies

The project uses Python and the following principal libraries:

- NumPy
- pandas
- Matplotlib
- Seaborn
- SciPy
- statsmodels
- ruptures
- openpyxl
- Jupyter and IPython kernel

Exact package versions are listed in `requirements.txt`.

## Limitations and Scope

- The dynamic-programming procedure minimizes ordinary regression residual sum of squares and does not explicitly model serial correlation in the segmentation objective.
- HAC covariance estimation adjusts the covariance matrix used for inference in the pre-specified-date test; it does not remove serial correlation from the underlying series or residuals.
- The April 2020 test evaluates a pre-specified date. It is not an exhaustive test of every possible break date.
- The variance-break simulation evaluates false coefficient-break classifications; it is not a dedicated variance-break detection experiment.
- The robustness experiment varies selected design factors for a single-intercept-break setting. It does not cover every possible break structure.
- The empirical conclusions depend on the selected CPI measure, sample period, transformations, regression specification, and minimum segment length.

## Reproducibility

The repository includes scripts, a notebook, processed data, simulation result files, tables, and figures to support inspection of the analysis. Random seeds, configuration settings, and output paths should be checked in the relevant scripts when reproducing or extending the experiments.

## Author

**Ananyo Sen**  
M.Sc. Student | Data Science
**Ayan Kumar Batabyal**  
M.Sc. Student | Data Science
**Sagnik Chandra**  
M.Sc. Student | Data Science


---

This project was developed as an academic study of structural-break testing, changepoint detection, simulation-based evaluation, and applied time-series analysis.
