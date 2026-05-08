# Sales Demand Forecasting


Predicting daily retail sales for 500 store-item combinations using classical time series methods.

---

## Project Overview

This project applies a complete time series forecasting pipeline to the [Kaggle Store-Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) dataset — 913,000 daily sales observations across 10 stores and 50 items spanning 5 years (2013–2017).

The goal is to forecast daily sales for all 500 store-item combinations 90 days into the future (January–March 2018) with 95% confidence intervals.

---

## Analysis Pipeline

1. Data loading and quality checks
2. Exploratory data analysis — trend, seasonality, rolling averages
3. STL decomposition — seasonal strength = 0.276
4. Stationarity testing — ADF + KPSS + ACF/PACF + differencing (d=1)
5. Model fitting — ARIMA, SARIMA, SARIMAX, ETS, Prophet
6. Residual diagnostics — Ljung-Box test on all models
7. Rolling origin cross-validation — 1,096 windows, horizons 1–30 days
8. K-means demand clustering — 4 archetypes from 6 demand features
9. 90-day production forecast — 45,000 predictions for all 500 combinations

---

## Key Findings

- **Prophet won** by capturing both the weekly day-of-week pattern AND the annual summer-winter cycle. That annual 10-unit swing is invisible to ARIMA and SARIMA — they see it only as noise.
- **ARIMA produces a flat forecast** — not a data issue, a structural property of ARIMA without seasonal terms. The AR memory fades in 4–5 days and every future prediction converges to zero change.
- **Weekend effect = +4.55 units/day** (p < 0.001) — the most actionable single finding from the SARIMAX coefficients.
- **ADF and KPSS contradicted each other** — resolved by trend-stationarity. The series has a deterministic trend but no stochastic unit root. d=1 differencing resolves both.
- **Store 2 needs 84% more inventory than Store 7** over the 90-day period — equal allocation across stores would systematically understock one and overstock the other.
- **ETS had the cleanest residuals** — Ljung-Box p=0.91, best of all five models — but not the best accuracy. Prophet wins on prediction; ETS wins on statistical diagnostics.

---

## Demand Clustering

K-means on 6 features (volume, volatility, trend, weekend lift, summer lift) identified 4 item archetypes:

| Cluster | Items | Avg Volume | CV | Recommended Policy |
|---|---|---|---|---|
| High Volume | 18 | 68.9/day | 0.274 | Lean inventory, frequent replenishment |
| Stable Mid-Volume | 17 | 44.8/day | 0.289 | Standard policy |
| Low Volume Stable | 9 | 24.7/day | 0.322 | Longer replenishment cycles |
| High Volatility | 6 | 23.2/day | 0.323 | Wide safety stock buffers |

---

## Repository Structure

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/vinamratha11/Demand-Forecasting-Analytics.git
cd Demand-Forecasting-Analytics
```

**2. Install R packages**
```r
install.packages(c("tidyverse", "lubridate", "forecast", "tseries",
                   "prophet", "ggplot2", "gridExtra", "Metrics", "zoo", "cluster"))
```

**3. Add the dataset**

Download `train.csv` and place it in the `dataset/` folder.

**4. Run the script**
```r
source("TimeSeries_DemandForecast_Code.R")
```

Runs end to end. Forecast CSVs are saved to `forecast_results/`. All plots are generated inline — save them manually to `output/` if needed.

---

## Tech Stack

| | Tool |
|---|---|
| Language | R 4.5.1 / RStudio |
| Data wrangling | tidyverse, lubridate |
| Time series modelling | forecast (v8.21) |
| Stationarity tests | tseries |
| Bayesian forecasting | prophet (Meta) |
| Accuracy metrics | Metrics |
| Clustering | cluster |
| Visualisation | ggplot2, gridExtra |

---

## Dataset

Source: [Kaggle — Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)

913,000 rows · 4 columns · 10 stores · 50 items · Jan 2013 – Dec 2017 · Zero missing dates


