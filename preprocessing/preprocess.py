"""Load the raw source data into DataFrames and run a quick EDA.

Run from the repo root:  python preprocessing/preprocess.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DELAY_THRESHOLD_MIN = 15  # DOT definition of a "delayed" flight
TEST_START = "2026-06-01"  # time-based split: train before, test from this date on


def load_raw() -> dict[str, pd.DataFrame]:
    """Load every raw dataset in /data into a dict of DataFrames."""
    frames = {
        "airlines": pd.read_csv(DATA_DIR / "airlines.csv", parse_dates=["first_flight_date", "last_flight_date"]),
        "airports": pd.read_csv(DATA_DIR / "airports.csv"),
        "daily_airport_stats": pd.read_csv(DATA_DIR / "daily_airport_stats.csv", parse_dates=["flight_date"]),
        "flights_sample": pd.read_csv(DATA_DIR / "flights_sample.csv", parse_dates=["flight_date"]),
    }
    return frames


def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}\n{name}: {df.shape[0]:,} rows x {df.shape[1]} cols\n{'=' * 80}")
    print("\n-- dtypes / nulls --")
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum(), "null_pct": (df.isna().mean() * 100).round(1)})
    print(summary.to_string())
    print("\n-- numeric describe --")
    print(df.describe().T.round(3).to_string())
    print(f"\nduplicate rows: {df.duplicated().sum():,}")


def explore(frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        summarize(name, df)

    fl = frames["flights_sample"]
    print("\n-- flights_sample: date range --", fl["flight_date"].min().date(), "to", fl["flight_date"].max().date())
    print("\n-- flights_sample: outcome rates --")
    print(fl[["dep_del15", "arr_del15", "cancelled", "diverted"]].mean().round(4).to_string())
    print("\n-- cancellation reasons --")
    print(fl["cancellation_reason"].value_counts(dropna=True).to_string())
    print("\n-- by airline (flights, dep_del15, arr_del15, cancelled) --")
    print(
        fl.groupby("airline")[["dep_del15", "arr_del15", "cancelled"]]
        .mean()
        .join(fl["airline"].value_counts().rename("flights"))
        .sort_values("flights", ascending=False)
        .round(4)
        .to_string()
    )
    print("\n-- delay rate by month / day_of_week / time block --")
    for col in ["month", "day_of_week", "dep_time_block"]:
        print(fl.groupby(col)["arr_del15"].mean().round(4).to_string(), "\n")
    print("\n-- delay-cause minutes (flights with arr_del15 == 1) --")
    causes = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
    print(fl.loc[fl["arr_del15"] == 1, causes].sum().div(fl.loc[fl["arr_del15"] == 1, causes].sum().sum()).round(3).to_string())

    ds = frames["daily_airport_stats"]
    print("\n-- daily_airport_stats --")
    print("date range:", ds["flight_date"].min().date(), "to", ds["flight_date"].max().date())
    print("airports:", ds["airport"].nunique(), "| rows/airport-day unique:", not ds.duplicated(["flight_date", "airport"]).any())

    # Cross-file consistency
    print("\n-- referential checks --")
    print("sample airlines not in airlines.csv:", set(fl["airline_code"]) - set(frames["airlines"]["airline_code"]))
    known = set(frames["airports"]["iata_code"])
    print("sample origins not in airports.csv:", len(set(fl["origin"]) - known))
    print("daily-stats airports not in airports.csv:", len(set(ds["airport"]) - known))


def build_training_set(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a flight-level modeling table from airline, date and airport information.

    Target: `delay_minutes` = the worse of departure and arrival delay (clipped at 0), so
    delays of any type/cause count. `log_delay_minutes` = log1p(delay_minutes) is the
    recommended regression target because delays are heavily right-skewed; `is_delayed`
    (>= 15 min) is kept for evaluating delay likelihood. Cancelled and diverted flights have
    no delay minutes and are dropped. Only information known before departure is used as
    features (airline, date, origin/destination, route distance, airport attributes); no
    post-flight columns or delay-cause columns are kept.
    """
    fl = frames["flights_sample"]
    fl = fl[(fl["cancelled"] == 0) & (fl["diverted"] == 0)].copy()
    fl = fl.dropna(subset=["dep_delay", "arr_delay"])

    # --- target ---
    fl["delay_minutes"] = np.maximum(fl["dep_delay"], fl["arr_delay"]).clip(lower=0)
    fl["log_delay_minutes"] = np.log1p(fl["delay_minutes"])
    fl["is_delayed"] = (fl["delay_minutes"] >= DELAY_THRESHOLD_MIN).astype(int)

    # --- date features ---
    d = fl["flight_date"]
    fl["day_of_month"] = d.dt.day
    fl["week_of_year"] = d.dt.isocalendar().week.astype(int)
    fl["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    holidays = USFederalHolidayCalendar().holidays(d.min() - pd.Timedelta(days=7), d.max() + pd.Timedelta(days=7))
    nearest = np.abs(d.values[:, None].astype("datetime64[D]") - holidays.values[None, :].astype("datetime64[D]")).min(axis=1)
    fl["days_to_nearest_holiday"] = nearest.astype("timedelta64[D]").astype(int)
    fl["near_holiday"] = (fl["days_to_nearest_holiday"] <= 2).astype(int)

    # --- airport features (origin_ / dest_ prefixed) ---
    ap = frames["airports"][["iata_code", "latitude", "longitude", "elevation_ft", "airport_type", "departures"]]
    for side, col in [("origin", "origin"), ("dest", "dest")]:
        renamed = ap.rename(columns={c: f"{side}_{c}" for c in ap.columns if c != "iata_code"}).rename(columns={"iata_code": col})
        fl = fl.merge(renamed, on=col, how="left", validate="m:1")
    # Airport volume is heavy-tailed too (3 to 573k departures): log-transform
    for side in ("origin", "dest"):
        fl[f"log_{side}_departures"] = np.log1p(fl.pop(f"{side}_departures"))
    fl["log_distance"] = np.log1p(fl["distance"])
    fl["route"] = fl["origin"] + "-" + fl["dest"]

    features = [
        "flight_date", "airline_code", "origin", "dest", "route",
        "month", "day_of_week", "day_of_month", "week_of_year", "is_weekend",
        "days_to_nearest_holiday", "near_holiday",
        "distance", "log_distance",
        "origin_latitude", "origin_longitude", "origin_elevation_ft", "origin_airport_type", "log_origin_departures",
        "dest_latitude", "dest_longitude", "dest_elevation_ft", "dest_airport_type", "log_dest_departures",
    ]
    targets = ["delay_minutes", "log_delay_minutes", "is_delayed"]
    return fl[features + targets].reset_index(drop=True)


def report_skew(df: pd.DataFrame) -> None:
    print("\n-- skewness before/after log transform --")
    pairs = [("delay_minutes", "log_delay_minutes"), ("distance", "log_distance")]
    for raw, logged in pairs:
        print(f"{raw}: skew {df[raw].skew():.2f} -> {logged}: skew {df[logged].skew():.2f}")
    print(f"share of flights with zero delay minutes: {(df['delay_minutes'] == 0).mean():.3f}")
    print(f"is_delayed rate: {df['is_delayed'].mean():.4f}")


def save_training_set(df: pd.DataFrame) -> None:
    """Write train/test parquet files, split by date so the test set is strictly later."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    test_mask = df["flight_date"] >= pd.Timestamp(TEST_START)
    train, test = df[~test_mask], df[test_mask]
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"\ntrain: {len(train):,} rows ({train['flight_date'].min().date()} to {train['flight_date'].max().date()})")
    print(f"test:  {len(test):,} rows ({test['flight_date'].min().date()} to {test['flight_date'].max().date()})")
    unseen = (set(test["airline_code"]) - set(train["airline_code"])) | (set(test["origin"]) - set(train["origin"]))
    print("test airlines/origins unseen in train:", unseen or "none")
    print(f"saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    frames = load_raw()
    explore(frames)
    training = build_training_set(frames)
    print(f"\n{'=' * 80}\ntraining set: {training.shape[0]:,} rows x {training.shape[1]} cols\n{'=' * 80}")
    print("nulls:", training.isna().sum()[lambda x: x > 0].to_dict() or "none")
    report_skew(training)
    save_training_set(training)
