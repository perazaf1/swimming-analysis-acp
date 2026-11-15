"""
Data cleaning and preparation pipeline for 100m Freestyle swimming analysis.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"



# --- Helpers for raw Top 200 and Olympic CSV parsing ---
def parse_swim_time(value: object) -> float:
    """Convert a text race time into decimal seconds, or return np.nan.

    Handled formats:
    - '46.86'           -> 46.86
    - '01:45.30'        -> 105.30
    - '00:01:00.000000' -> 60.0
    - '22:36.4est'      -> 1356.4  (estimated historical time)
    - 'Disqualified', 'Did not start', '', None, np.nan -> np.nan
    """
    if pd.isna(cast(Any, value)):
        return np.nan

    text = str(value).strip()
    if text == "":
        return np.nan

    # Older Olympic entries sometimes append "est" for hand-timed estimates.
    if text.endswith("est"):
        text = text[:-3].strip()

    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return np.nan

    if len(parts) == 1:
        return round(parts[0], 4)
    if len(parts) == 2:
        minutes, seconds = parts
        return round(60 * minutes + seconds, 4)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return round(3600 * hours + 60 * minutes + seconds, 4)
    return np.nan


def parse_two_digit_date(series: pd.Series, pivot_year: int = 25) -> pd.Series:
    """Parse MM-DD-YY strings and shift pre-pivot 20xx dates back to 19xx."""
    dates = pd.to_datetime(series, format="%m-%d-%y", errors="coerce")
    # pandas maps some 2-digit years above its cutoff to 20xx instead of 19xx.
    mask = dates.dt.year > 2000 + pivot_year
    dates.loc[mask] = dates.loc[mask] - pd.DateOffset(years=100)
    return dates


def normalize_athlete_name(name: object) -> str:
    """Standardize 'POPOVICI, David' into 'David Popovici' for cross-file joins."""
    if pd.isna(cast(Any, name)):
        return ""
    text = " ".join(str(name).strip().split())
    if "," in text:
        last, first = text.split(",", maxsplit=1)
        text = f"{first.strip()} {last.strip()}".strip()
    return text.title()


def clean_top_perf(path: Path = RAW_DIR / "top_perf_categories.csv") -> pd.DataFrame:
    """Load and clean the World Aquatics Top 200 dataset."""
    df = pd.read_csv(path, encoding="utf-8-sig", sep=",", usecols=lambda col: col != "index")

    df["swim_time_s"] = df["Swim time"].map(parse_swim_time)
    df["swim_date"] = parse_two_digit_date(df["Swim date"])
    df["birth_date"] = parse_two_digit_date(df["Athlete birth date"])
    df["swim_year"] = df["swim_date"].dt.year
    df["age"] = (df["swim_date"] - df["birth_date"]).dt.days / 365.25

    df["gender"] = df["Gender"].map({"M": "Men", "F": "Women"})
    df["athlete_name"] = df["Athlete Full Name"].map(normalize_athlete_name)

    def _normalize_event(desc: str) -> str:
        # Format in Event description is "<Gender> <Dist> <Stroke> LCM <Sex>".
        _, dist, stroke, *_ = desc.strip().split()
        if stroke == "Medley":
            stroke = "Individual Medley"
        return f"{dist}m {stroke}"

    df["event"] = df["Event description"].map(_normalize_event)
    df["rank_order"] = pd.to_numeric(df["Rank_Order"], errors="coerce")

    df = df.rename(
        columns={
            "Team Code": "team_code",
            "Event Name": "competition",
            "City": "city",
            "Country Code": "country_code",
        }
    )

    cols = [
        "event",
        "gender",
        "athlete_name",
        "team_code",
        "swim_time_s",
        "swim_date",
        "swim_year",
        "birth_date",
        "age",
        "rank_order",
        "competition",
        "city",
        "country_code",
    ]
    return df[cols]



def build_longitudinal_100m_table(df_top: pd.DataFrame) -> pd.DataFrame:
    """Build the 100m Freestyle Top 200 table with 200m Free and 100m Fly personal bests."""
    df_100 = df_top[df_top["event"] == "100m Freestyle"].copy()

    # Attach each swimmer's fastest recorded 200m Free and 100m Fly times when present.
    for ev, col_name in [
        ("200m Freestyle", "pb_200m_freestyle_s"),
        ("100m Butterfly", "pb_100m_butterfly_s"),
    ]:
        pbs = (
            df_top[df_top["event"] == ev]
            .groupby(["gender", "athlete_name"], as_index=False)[["swim_time_s"]]
            .min()
            .rename(columns={"swim_time_s": col_name})
        )
        df_100 = df_100.merge(pbs, on=["gender", "athlete_name"], how="left")

    # Track repeated Top 200 appearances (especially dense on the men's side).
    grouped = df_100.groupby(["gender", "athlete_name"])["swim_time_s"]
    df_100["pb_100m_freestyle_s"] = grouped.transform("min")
    df_100["n_appearances_100m"] = grouped.transform("count")

    df_100 = df_100.sort_values(["gender", "swim_time_s"]).reset_index(drop=True)
    return df_100


def clean_olympic_results(
    path: Path = RAW_DIR / "Results_1912to2020.csv",
    min_year: int | None = 1988,
) -> pd.DataFrame:
    """Load and clean individual Olympic swimming results (1988+ by default)."""
    df_olympic = pd.read_csv(path, encoding="utf-8-sig", sep=",")
    df_olympic = df_olympic[df_olympic["Relay?"] == 0].copy()
    df_olympic = df_olympic[
        df_olympic["Athlete"].notna()
        & (df_olympic["Athlete"].astype(str).str.strip() != "")
    ]

    df_olympic["swim_time_s"] = df_olympic["Results"].map(parse_swim_time)
    df_olympic = df_olympic.dropna(subset=["swim_time_s"]).copy()

    df_olympic["year"] = pd.to_numeric(df_olympic["Year"], errors="coerce").astype("Int64")
    # 50m Freestyle became a permanent Olympic event in 1988.
    if min_year is not None:
        df_olympic = df_olympic[df_olympic["year"] >= min_year].copy()

    df_olympic["gender"] = df_olympic["Gender"].astype(str).str.strip()
    df_olympic["athlete_name"] = df_olympic["Athlete"].map(normalize_athlete_name)
    stroke = (
        df_olympic["Stroke"]
        .astype(str)
        .str.strip()
        .replace({"Individual medley": "Individual Medley"})
    )
    df_olympic["event"] = df_olympic["Distance (in meters)"].astype(str).str.strip() + " " + stroke
    df_olympic["rank"] = pd.to_numeric(df_olympic["Rank"], errors="coerce")

    df_olympic = df_olympic.rename(
        columns={
            "Team": "team_code",
            "Location": "location",
        }
    )

    cols = [
        "event",
        "gender",
        "athlete_name",
        "team_code",
        "year",
        "location",
        "swim_time_s",
        "rank",
    ]
    return df_olympic[cols].drop_duplicates().reset_index(drop=True)


DEFAULT_MULTI_EVENTS: tuple[str, ...] = (
    "50m Freestyle",
    "100m Freestyle",
    "200m Freestyle",
    "400m Freestyle",
    "100m Butterfly",
    "200m Individual Medley",
)


def build_multi_event_matrix(
    df_olympic: pd.DataFrame,
    df_top: pd.DataFrame | None = None,
    events: tuple[str, ...] = DEFAULT_MULTI_EVENTS,
) -> pd.DataFrame:
    """Pivot swimmer personal bests into an athlete x events matrix."""
    cols = ["gender", "athlete_name", "event", "swim_time_s"]
    frames = [df_olympic[cols]]
    if df_top is not None:
        frames.append(df_top[cols])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["event"].isin(events)]

    # Keep only each athlete's personal best when merging Olympic and Top 200 data.
    combined = (
        combined.groupby(["gender", "athlete_name", "event"], as_index=False)[["swim_time_s"]]
        .min()
    )

    matrix = combined.pivot(index=["gender", "athlete_name"], columns="event", values="swim_time_s")
    ordered_cols = [ev for ev in events if ev in matrix.columns]
    matrix = matrix[ordered_cols].reset_index()
    matrix.columns.name = None

    matrix["n_events"] = matrix[ordered_cols].notna().sum(axis=1)
    matrix = matrix.sort_values(
        ["n_events", "gender", "athlete_name"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return matrix


def run_phase1_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    olympic_min_year: int | None = 1988,
) -> dict[str, pd.DataFrame]:
    """Run the full cleaning pipeline and write all processed CSV tables."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_top = clean_top_perf(raw_dir / "top_perf_categories.csv")
    df_100 = build_longitudinal_100m_table(df_top)
    df_oly = clean_olympic_results(raw_dir / "Results_1912to2020.csv", min_year=olympic_min_year)
    df_multi = build_multi_event_matrix(df_oly, df_top)

    outputs = {
        "top_perf_cleaned": df_top,
        "longitudinal_100m_nl": df_100,
        "olympic_results_cleaned": df_oly,
        "multi_event_matrix": df_multi,
    }

    for name, frame in outputs.items():
        frame.to_csv(processed_dir / f"{name}.csv", index=False)

    return outputs


if __name__ == "__main__":
    # 1. Unit tests: time parsing
    time_tests = [
        ("46.86", 46.86),
        ("01:45.30", 105.30),
        ("00:01:00.000000", 60.0),
        ("22:36.4est", 1356.4),
    ]
    for raw, expected in time_tests:
        assert parse_swim_time(raw) == expected, f"Failed for {raw!r}"
    for raw in ("Disqualified", "", None):
        assert np.isnan(parse_swim_time(raw)), f"Expected NaN for {raw!r}"

    # 2. Unit tests: 2-digit year pivot
    dates_sample = pd.Series(["10-27-64", "09-15-04", "01-10-87", "invalid", None])
    parsed_dates = parse_two_digit_date(dates_sample)
    assert parsed_dates.iloc[0].year == 1964
    assert parsed_dates.iloc[1].year == 2004
    assert parsed_dates.iloc[2].year == 1987
    assert pd.isna(parsed_dates.iloc[3]) and pd.isna(parsed_dates.iloc[4])

    # 3. Unit tests: athlete name normalization
    assert normalize_athlete_name("POPOVICI, David") == "David Popovici"
    assert normalize_athlete_name("  cielo filho,  Cesar ") == "Cesar Cielo Filho"
    assert normalize_athlete_name("Evgeny Rylov") == "Evgeny Rylov"
    assert normalize_athlete_name(None) == ""

    # 4. Integration checks on real datasets
    df_top = clean_top_perf()
    assert df_top.shape == (5200, 13), f"Unexpected shape: {df_top.shape}"
    for col in ("swim_time_s", "swim_date", "birth_date", "age", "gender", "event"):
        assert df_top[col].isna().sum() == 0, f"Unexpected NaNs in {col}"
    assert 13.0 < df_top["age"].min() and df_top["age"].max() < 40.0
    assert {"100m Freestyle", "200m Individual Medley"}.issubset(df_top["event"].unique())
    assert df_top.loc[df_top["event"] == "100m Freestyle", "swim_time_s"].min() == 46.86

    df_100 = build_longitudinal_100m_table(df_top)
    assert df_100 is not None
    assert df_100.shape == (400, 17), f"Unexpected shape: {df_100.shape}"
    assert list(df_100.index) == list(range(400))
    assert (df_100["event"] == "100m Freestyle").all()
    assert df_100.iloc[0]["athlete_name"] == "David Popovici"
    assert df_100.iloc[0]["n_appearances_100m"] == 15
    assert pd.notna(df_100.iloc[0]["pb_200m_freestyle_s"])
    assert df_100["pb_100m_butterfly_s"].notna().sum() > 0

    df_oly = clean_olympic_results()
    assert df_oly.duplicated().sum() == 0
    assert df_oly.shape == (1869, 8), f"Unexpected shape: {df_oly.shape}"
    assert list(df_oly.index) == list(range(len(df_oly)))
    assert df_oly["year"].min() == 1988 and df_oly["year"].max() == 2020
    assert {"50m Freestyle", "100m Freestyle", "200m Individual Medley"}.issubset(
        df_oly["event"].unique()
    )
    assert df_oly["swim_time_s"].isna().sum() == 0

    df_multi = build_multi_event_matrix(df_oly, df_top)
    assert df_multi.shape == (726, 9), f"Unexpected shape: {df_multi.shape}"
    assert list(df_multi.index) == list(range(len(df_multi)))
    assert df_multi["n_events"].max() == 4
    assert df_multi.iloc[0]["n_events"] == 4
    assert df_multi["n_events"].min() >= 1

    outputs = run_phase1_pipeline()
    assert outputs is not None
    for name in outputs:
        out_path = PROCESSED_DIR / f"{name}.csv"
        assert out_path.exists() and out_path.stat().st_size > 0, f"Missing file: {out_path}"

    print("All cleaning checks passed.")
    print(f"top_perf shape: {df_top.shape} | age min/max: {df_top['age'].min():.2f} / {df_top['age'].max():.2f} yrs")
    print(f"longitudinal_100m shape: {df_100.shape}")
    print(f"olympic_results shape: {df_oly.shape} | years: {df_oly['year'].min()}-{df_oly['year'].max()}")
    print(
        f"multi_event_matrix shape: {df_multi.shape} | "
        f">= 2 events: {(df_multi['n_events'] >= 2).sum()} | "
        f">= 3 events: {(df_multi['n_events'] >= 3).sum()}"
    )
    print("Generated files in data/processed/:", [f"{k}.csv" for k in outputs])
    print(df_multi.head(5))