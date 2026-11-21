"""Load the raw source data into DataFrames and run a quick EDA.

Run from the repo root:  python preprocessing/preprocess.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    explore(load_raw())
