"""Case 3 — temporal features derived from TransactionDT.

TransactionDT is a seconds-elapsed offset from an undocumented reference point, not a real
timestamp (established in Case 1). day_of_week_proxy and is_weekend_proxy anchor day 0 to
2017-12-01 (a Friday) — not an official value from the competition organizers, but a date
independently arrived at both by web research done in this session (at least one public IEEE-CIS
analysis uses the same anchor when converting TransactionDT to calendar time) and by separate
user research (derived from US holiday-season transaction-volume patterns — the Black
Friday/Cyber Monday spike and the Christmas-period shift). Treated here as a well-reasoned but
unconfirmed community estimate, not ground truth: column names keep the `_proxy` suffix
deliberately, and the cyclical (sin/cos) encodings a model would actually consume are computed
from the raw weekday phase number, independent of which real weekday name got attached to it.

hour_of_day / day_of_period reuse the exact `// 3600 % 24` / `// 86400` arithmetic Case 1's
distributions.py::analyze_time_distribution already used — same source of truth, not re-derived
differently in a second place.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600

# Community-estimated reference date for TransactionDT == 0 — see module docstring. Not
# officially confirmed by the competition organizers (Vesta/IEEE).
REFERENCE_DATE = date(2017, 12, 1)
REFERENCE_WEEKDAY_INDEX = REFERENCE_DATE.weekday()  # Monday=0 .. Sunday=6; 4 == Friday, verified

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKEND_DAY_NAMES = {"Saturday", "Sunday"}

# Case 1, section 6: the hour band with the lowest volume and highest fraud rate (hour 7: 10.61%).
LOW_VOLUME_HOUR_START = 4
LOW_VOLUME_HOUR_END = 9  # exclusive


def _cyclical_encode(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_temporal_features(parquet_path: Path) -> pd.DataFrame:
    """Returns TransactionID + temporal feature columns, ready to join back onto the main
    dataset — the merged parquet itself is never modified, feature sets stay a separate layer."""
    df = pq.ParquetFile(parquet_path).read(columns=["TransactionID", "TransactionDT"]).to_pandas()
    dt = df["TransactionDT"].to_numpy()

    hour_of_day = (dt // SECONDS_PER_HOUR) % 24
    day_of_period = dt // SECONDS_PER_DAY
    weekday_index = (REFERENCE_WEEKDAY_INDEX + day_of_period) % 7  # Monday=0 .. Sunday=6

    hour_sin, hour_cos = _cyclical_encode(hour_of_day, 24)
    dow_sin, dow_cos = _cyclical_encode(weekday_index, 7)

    weekday_names = np.array(WEEKDAY_NAMES)[weekday_index]
    is_weekend = np.isin(weekday_names, list(WEEKEND_DAY_NAMES))
    is_low_volume_hour = (hour_of_day >= LOW_VOLUME_HOUR_START) & (hour_of_day < LOW_VOLUME_HOUR_END)

    return pd.DataFrame({
        "TransactionID": df["TransactionID"],
        "hour_of_day": hour_of_day,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_period": day_of_period,
        "day_of_week_proxy": weekday_names,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "is_weekend_proxy": is_weekend,
        "is_low_volume_hour": is_low_volume_hour,
    })
