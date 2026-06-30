from __future__ import annotations

import numpy as np
import pandas as pd


def latest_observed(row: pd.Series) -> tuple[int, float]:
    observed = row.dropna()
    if observed.empty:
        return -1, np.nan
    return int(observed.index[-1]), float(observed.iloc[-1])


def volume_weighted_factors(cumulative_triangle: pd.DataFrame) -> pd.Series:
    factors = {}
    cols = list(cumulative_triangle.columns)
    for current_dev, next_dev in zip(cols[:-1], cols[1:]):
        current = cumulative_triangle[current_dev]
        next_values = cumulative_triangle[next_dev]
        mask = current.notna() & next_values.notna() & (current > 0)
        if mask.sum() == 0:
            factors[current_dev] = 1.0
        else:
            raw = next_values[mask].sum() / current[mask].sum()
            factors[current_dev] = max(float(raw), 1.0)
    return pd.Series(factors, name="selected_factor")


def age_to_ultimate_factors(selected_factors: pd.Series, columns: list[int]) -> pd.Series:
    atu = {}
    for dev in columns:
        future = selected_factors[selected_factors.index >= dev]
        factor = float(future.prod()) if len(future) else 1.0
        atu[dev] = max(factor, 1.0)
    return pd.Series(atu, name="age_to_ultimate")


def project_chain_ladder_with_factors(
    cumulative_triangle: pd.DataFrame,
    selected_factors: pd.Series,
    age_to_ultimate: pd.Series,
) -> pd.DataFrame:
    rows = []
    for ay, row in cumulative_triangle.iterrows():
        latest_dev, latest_value = latest_observed(row)
        atu_factor = float(age_to_ultimate.get(latest_dev, 1.0))
        ultimate = latest_value * atu_factor if pd.notna(latest_value) else np.nan
        reserve = ultimate - latest_value if pd.notna(ultimate) else np.nan
        rows.append(
            {
                "Accident Year": ay,
                "Latest Development": latest_dev,
                "Latest Cumulative": latest_value,
                "Age-to-Ultimate Factor": atu_factor,
                "Ultimate Loss": ultimate,
                "Reserve": max(reserve, 0.0) if pd.notna(reserve) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def chain_ladder(cumulative_triangle: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    factors = volume_weighted_factors(cumulative_triangle)
    atu = age_to_ultimate_factors(factors, list(cumulative_triangle.columns))
    return factors, atu, project_chain_ladder_with_factors(cumulative_triangle, factors, atu)


def expected_loss_ratio(
    cumulative_triangle: pd.DataFrame,
    exposure: pd.DataFrame | None = None,
    expected_loss_ratio_value: float = 0.72,
) -> pd.DataFrame:
    latest = []
    latest_by_year = {}
    for ay, row in cumulative_triangle.iterrows():
        _, value = latest_observed(row)
        latest_by_year[int(ay)] = value

    if exposure is not None and not exposure.empty and {"Policy year", "Exposure"}.issubset(exposure.columns):
        exp = exposure.rename(columns={"Policy year": "Accident Year"}).copy()
        exp["Accident Year"] = pd.to_numeric(exp["Accident Year"], errors="coerce")
        exp["Exposure"] = pd.to_numeric(exp["Exposure"], errors="coerce")
        exp = exp.dropna()
        scale_years = [int(y) for y in cumulative_triangle.index if int(y) in set(exp["Accident Year"])]
        observed = np.array([latest_by_year[y] for y in scale_years if pd.notna(latest_by_year[y])], dtype=float)
        exposure_values = np.array(
            [exp.loc[exp["Accident Year"] == y, "Exposure"].iloc[0] for y in scale_years if pd.notna(latest_by_year[y])],
            dtype=float,
        )
        positive = exposure_values > 0
        if positive.any() and np.nansum(observed[positive]) > 0:
            pure_premium = float(np.nansum(observed[positive]) / np.nansum(exposure_values[positive]))
        else:
            pure_premium = float(np.nanmean(list(latest_by_year.values())))
        exposure_map = exp.set_index("Accident Year")["Exposure"].to_dict()
    else:
        pure_premium = float(np.nanmean([v for v in latest_by_year.values() if pd.notna(v)]))
        exposure_map = {ay: 1.0 for ay in latest_by_year}

    for ay in cumulative_triangle.index:
        ay_int = int(ay)
        latest_value = latest_by_year[ay_int]
        exposure_value = float(exposure_map.get(ay_int, 1.0))
        expected_ultimate = pure_premium * exposure_value * expected_loss_ratio_value
        if expected_ultimate < latest_value:
            expected_ultimate = latest_value
        latest.append(
            {
                "Accident Year": ay_int,
                "Exposure": exposure_value,
                "Expected Loss Ratio": expected_loss_ratio_value,
                "Expected Ultimate Loss": expected_ultimate,
                "Latest Cumulative": latest_value,
                "Reserve": max(expected_ultimate - latest_value, 0.0),
            }
        )
    return pd.DataFrame(latest)


def bornhuetter_ferguson(
    cumulative_triangle: pd.DataFrame,
    age_to_ultimate: pd.Series,
    expected_ultimate: pd.DataFrame,
) -> pd.DataFrame:
    expected_map = expected_ultimate.set_index("Accident Year")["Expected Ultimate Loss"].to_dict()
    rows = []
    for ay, row in cumulative_triangle.iterrows():
        latest_dev, latest_value = latest_observed(row)
        atu = float(age_to_ultimate.get(latest_dev, 1.0))
        percent_reported = 1.0 / atu if atu > 0 else 1.0
        unreported_percent = max(1.0 - percent_reported, 0.0)
        prior_ultimate = float(expected_map.get(int(ay), latest_value))
        reserve = prior_ultimate * unreported_percent
        ultimate = latest_value + reserve
        rows.append(
            {
                "Accident Year": int(ay),
                "Latest Development": latest_dev,
                "Latest Cumulative": latest_value,
                "Prior Ultimate Loss": prior_ultimate,
                "Percent Reported": percent_reported,
                "BF Reserve": max(reserve, 0.0),
                "BF Ultimate Loss": ultimate,
            }
        )
    return pd.DataFrame(rows)
