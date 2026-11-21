# Predicting Airline Reliability 

## Navigating the Repo
`/data` contains the original source data in 5 files:
1. `data/airlines.csv` aggregates delay and cancellation data at the airline level
2. `data/airports.csv` aggregates delay and cancellation data at the airport level
3. `data/daily_airport_stats.csv` provides a daily snapshot of flight delay/cancellation/departure aggregates by day by airport
4. `data/flights_sample.csv` contains the dataset of departure/delay/cancellatio data aggregated at the level of the individual flights


## Source Data Analysis
Initial exploratory analysis lives in `preprocessing/preprocess.py`, which loads each raw file in `/data` into a pandas DataFrame and prints summaries (run `python preprocessing/preprocess.py` from the repo root).

### Dataset overview
| File | Rows x Cols | Grain | Notes |
|---|---|---|---|
| `airlines.csv` | 14 x 7 | airline | ~11.7M total flights; no nulls or duplicates |
| `airports.csv` | 361 x 11 | airport | 99 large, 259 medium, 3 small; 52 states/territories; no nulls |
| `daily_airport_stats.csv` | 202,872 x 13 | airport-day | 361 airports; (date, airport) is unique |
| `flights_sample.csv` | 500,000 x 40 | flight | ~4% sample of the full flight-level data |

All flight-level data covers **2025-01-01 to 2026-08-31**. Keys are consistent across files: every airline code and every origin/destination airport in the flight sample and daily stats appears in `airlines.csv` / `airports.csv`.

### Key findings
- **Overall reliability (flight sample):** 22.7% of flights depart 15+ minutes late, 22.8% arrive 15+ minutes late, 1.8% are cancelled and 0.3% are diverted.
- **Airline differences are meaningful.** Arrival-delay rates range from 17.3% (Hawaiian) to ~28% (Frontier, PSA). Cancellation rates range from 0.6% (Allegiant) to 4.9% (PSA) and 4.2% (Republic). Large carriers like Delta, United and Alaska sit near 20-22% late arrivals.
- **Delays have strong time patterns.**
  - Time of day: 9% late for departures before 6am, rising steadily to ~33% in the 8-9pm block.
  - Season: peaks in June/July (27-29%) and December (27%); lowest in September (17%).
  - Day of week: Sunday (26%) and Monday (25%) are worst; Tuesday (19%) and Wednesday (20%) are best.
- **Delay causes** (share of delay minutes on delayed flights): late-arriving aircraft 39%, carrier 33%, NAS 21%, weather 7%, security <1%. Weather, however, drives most cancellations (5,720 of 8,884 cancelled flights in the sample).
- **Daily airport stats** are highly skewed: median airport-day has 9 scheduled departures (max 1,161), and `dep_delay_mean` ranges from -65 to 1,614 minutes, so small-airport days are noisy.

### Data quality notes and caveats
- **Structural nulls, not errors.** ~1.8-2.1% of flight-level timing/delay fields are null because the flight was cancelled or diverted. The five delay-cause columns (`carrier_delay`, etc.) are null for 77.7% of rows - i.e. any flight that was not 15+ minutes late - and should be filled with 0 or handled explicitly. `cancellation_code`/`cancellation_reason` are null for the 98.2% of flights that were not cancelled.
- **Daily stats nulls:** ~0.4% of airport-days have null delay rates (all flights cancelled that day).
- **Possible label oddities:** 398 cancelled flights in the sample still have a `dep_time` (likely cancelled after pushback), and one flight has a non-positive `sched_elapsed_time`.
- **Heavy-tailed delays:** `dep_delay` and `arr_delay` reach 3,400+ minutes, so use robust statistics or clip when modeling.
- **Time fields are HHMM integers** (e.g. `25` = 00:25, `2400` appears in actuals) rather than timestamps and need conversion.
- **Incomplete airline coverage:** Hawaiian's last flight is 2025-12-31 and Spirit's is 2026-05-01 (the other 12 run through 2026-08-31), so these carriers have shorter histories.
- **Non-contiguous US airports:** `airports.csv` includes territories (latitudes down to -14, longitudes up to +145), which may need handling for geographic features.
- **Leakage risk:** `arr_delay`, `arr_del15`, the delay-cause columns, `cancelled` and `diverted` are only known after a flight, so they must be excluded from features when predicting reliability in advance.



## Training Set Preprocessing
`preprocessing/preprocess.py` also builds the modeling table (`build_training_set`) and writes `data/processed/train.parquet` and `data/processed/test.parquet` (git-ignored; regenerate with `python preprocessing/preprocess.py`). It currently reads `flights_sample.csv` and will use the full `flights` data if it is loaded.

- **Unit and filtering:** one row per flight. Cancelled and diverted flights are dropped (they have no delay minutes), leaving 489,636 rows.
- **Target:** `delay_minutes` is the worse of departure and arrival delay, clipped at 0, so any delay cause counts. 52% of flights have zero delay minutes. `is_delayed` flags 15+ minutes (27.2% of flights).
- **Log transform:** the raw target is extremely right-skewed (skewness 10.9). `log_delay_minutes = log1p(delay_minutes)` brings skewness to 0.76 and is the recommended regression target. Predictions can be mapped back with `expm1`. `distance` (1.4 to -0.35) and airport volume are also log-transformed.
- **Features (all known before departure):**
  - Airline and route: `airline_code`, `origin`, `dest`, `route`
  - Date: `month`, `day_of_week`, `day_of_month`, `week_of_year`, `is_weekend`, `days_to_nearest_holiday`, `near_holiday`
  - Distance: `distance`, `log_distance`
  - Airport attributes from `airports.csv`: latitude, longitude, elevation, airport type, log departure volume
- **Excluded to avoid leakage:** actual times, taxi/air time, arrival fields and the delay-cause columns. Scheduled departure time is not included because the model is specified in terms of airline, date and airports, but it is a strong predictor (9% late before 6am vs 33% at 8pm) and worth adding later.
- **Split:** time-based, so the model is tested on data later than anything it trained on. Train is 2025-01-01 to 2026-05-31 (413,112 rows) and test is 2026-06-01 to 2026-08-31 (76,524 rows).
- **Still to do at modeling time:** one-hot or target-encode the categorical columns (`route` has high cardinality). Four test-set origins (ERI, LNS, DJT, HEZ) never appear in train, so the encoder must handle unseen categories.
