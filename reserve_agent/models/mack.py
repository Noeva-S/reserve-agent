from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MackResult:
    by_year: pd.DataFrame
    diagnostics: dict[str, float]


def _latest_observed(row: pd.Series) -> tuple[int, float]:
    observed = row.dropna()
    if observed.empty:
        return -1, np.nan
    return int(observed.index[-1]), float(observed.iloc[-1])


def _ordered_columns(cumulative_triangle: pd.DataFrame) -> list[int]:
    return [int(col) for col in cumulative_triangle.columns]


def _development_statistics(
    cumulative_triangle: pd.DataFrame,
    selected_factors: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Estimate Mack variance parameters for each development age.

    The implementation follows the standard Mack variance estimator where
    enough paired observations exist. Sparse ages fall back to the nearest
    credible variance so the app can still produce a stable teaching output.
    """

    cols = list(cumulative_triangle.columns)
    sigma_squared: dict[int, float] = {}
    exposure_by_age: dict[int, float] = {}
    pair_count_by_age: dict[int, float] = {}

    for current_dev, next_dev in zip(cols[:-1], cols[1:]):
        current = pd.to_numeric(cumulative_triangle[current_dev], errors="coerce")
        next_values = pd.to_numeric(cumulative_triangle[next_dev], errors="coerce")
        mask = current.notna() & next_values.notna() & (current > 0)
        pair_count = int(mask.sum())
        pair_count_by_age[int(current_dev)] = float(pair_count)
        exposure_by_age[int(current_dev)] = float(current[mask].sum()) if pair_count else 0.0

        factor = float(selected_factors.get(current_dev, selected_factors.get(int(current_dev), 1.0)))
        if pair_count >= 2:
            ratios = next_values[mask] / current[mask]
            weighted_errors = current[mask] * (ratios - factor) ** 2
            sigma_squared[int(current_dev)] = max(float(weighted_errors.sum() / (pair_count - 1)), 0.0)
        elif pair_count == 1:
            ratio = float((next_values[mask] / current[mask]).iloc[0])
            sigma_squared[int(current_dev)] = max(float(current[mask].iloc[0] * (ratio - factor) ** 2), 0.0)
        else:
            sigma_squared[int(current_dev)] = 0.0

    positive = [value for value in sigma_squared.values() if value > 0]
    fallback = float(np.nanmedian(positive)) if positive else 0.0
    for dev, value in list(sigma_squared.items()):
        if value == 0.0 and pair_count_by_age.get(dev, 0.0) < 2:
            sigma_squared[dev] = fallback

    return (
        pd.Series(sigma_squared, name="mack_sigma_squared"),
        pd.Series(exposure_by_age, name="mack_exposure"),
        pd.Series(pair_count_by_age, name="mack_pair_count"),
    )


def _project_cumulative_at_age(
    latest_value: float,
    latest_dev: int,
    target_dev: int,
    selected_factors: pd.Series,
) -> float:
    value = float(latest_value)
    for dev in selected_factors.index:
        dev_int = int(dev)
        if latest_dev <= dev_int < target_dev:
            value *= float(selected_factors.loc[dev])
    return value


def _reserve_interval(reserve: float, standard_error: float, z_value: float) -> tuple[float, float]:
    if pd.isna(reserve) or pd.isna(standard_error):
        return np.nan, np.nan
    lower = max(float(reserve) - z_value * float(standard_error), 0.0)
    upper = max(float(reserve) + z_value * float(standard_error), 0.0)
    return lower, upper


def mack_chain_ladder(
    cumulative_triangle: pd.DataFrame,
    selected_factors: pd.Series,
    age_to_ultimate: pd.Series,
) -> MackResult:
    """Run a robust Mack Chain Ladder approximation on a cumulative triangle."""

    cols = _ordered_columns(cumulative_triangle)
    if len(cols) < 2:
        empty = pd.DataFrame(
            columns=[
                "Accident Year",
                "Latest Development",
                "Latest Cumulative",
                "Age-to-Ultimate Factor",
                "Mack Ultimate",
                "Mack Reserve",
                "Mack Standard Error",
                "Mack CV",
                "Mack 75% Lower",
                "Mack 75% Upper",
                "Mack 95% Lower",
                "Mack 95% Upper",
            ]
        )
        return MackResult(empty, _diagnostics(empty, 0.0))

    sigma_squared, exposure_by_age, _ = _development_statistics(cumulative_triangle, selected_factors)
    terminal_dev = max(cols)
    rows: list[dict[str, float | int]] = []

    for ay, row in cumulative_triangle.iterrows():
        latest_dev, latest_value = _latest_observed(row)
        if latest_dev < 0 or pd.isna(latest_value):
            rows.append(
                {
                    "Accident Year": int(ay),
                    "Latest Development": latest_dev,
                    "Latest Cumulative": np.nan,
                    "Age-to-Ultimate Factor": np.nan,
                    "Mack Ultimate": np.nan,
                    "Mack Reserve": np.nan,
                    "Mack Standard Error": np.nan,
                    "Mack CV": np.nan,
                    "Mack 75% Lower": np.nan,
                    "Mack 75% Upper": np.nan,
                    "Mack 95% Lower": np.nan,
                    "Mack 95% Upper": np.nan,
                }
            )
            continue

        atu_factor = float(age_to_ultimate.get(latest_dev, 1.0))
        ultimate = float(latest_value) * atu_factor
        reserve = max(ultimate - float(latest_value), 0.0)
        variance_sum = 0.0
        for dev in selected_factors.index:
            dev_int = int(dev)
            if not (latest_dev <= dev_int < terminal_dev):
                continue
            factor = max(float(selected_factors.loc[dev]), 1e-12)
            sigma2 = max(float(sigma_squared.get(dev_int, 0.0)), 0.0)
            predicted_at_age = max(
                _project_cumulative_at_age(float(latest_value), latest_dev, dev_int, selected_factors),
                1e-12,
            )
            observed_exposure = max(float(exposure_by_age.get(dev_int, 0.0)), 1e-12)
            variance_sum += (sigma2 / (factor**2)) * ((1.0 / predicted_at_age) + (1.0 / observed_exposure))

        mse = max((ultimate**2) * variance_sum, 0.0)
        standard_error = float(np.sqrt(mse))
        cv_base = reserve if reserve > 0 else ultimate
        cv = standard_error / cv_base if cv_base > 0 else 0.0
        lower75, upper75 = _reserve_interval(reserve, standard_error, 1.150349)
        lower95, upper95 = _reserve_interval(reserve, standard_error, 1.959964)

        rows.append(
            {
                "Accident Year": int(ay),
                "Latest Development": latest_dev,
                "Latest Cumulative": float(latest_value),
                "Age-to-Ultimate Factor": atu_factor,
                "Mack Ultimate": ultimate,
                "Mack Reserve": reserve,
                "Mack Standard Error": standard_error,
                "Mack CV": cv,
                "Mack 75% Lower": lower75,
                "Mack 75% Upper": upper75,
                "Mack 95% Lower": lower95,
                "Mack 95% Upper": upper95,
            }
        )

    result = pd.DataFrame(rows)
    aggregate_se = _aggregate_standard_error(result, selected_factors, sigma_squared, exposure_by_age)
    return MackResult(result, _diagnostics(result, aggregate_se))


def _aggregate_standard_error(
    result: pd.DataFrame,
    selected_factors: pd.Series,
    sigma_squared: pd.Series,
    exposure_by_age: pd.Series,
) -> float:
    if result.empty:
        return 0.0

    individual_variance = float((pd.to_numeric(result["Mack Standard Error"], errors="coerce") ** 2).sum())
    covariance = 0.0
    rows = result.dropna(subset=["Mack Ultimate", "Latest Development"]).reset_index(drop=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            row_i = rows.iloc[i]
            row_j = rows.iloc[j]
            shared_variance = 0.0
            start_dev = max(int(row_i["Latest Development"]), int(row_j["Latest Development"]))
            for dev in selected_factors.index:
                dev_int = int(dev)
                if dev_int < start_dev:
                    continue
                factor = max(float(selected_factors.loc[dev]), 1e-12)
                sigma2 = max(float(sigma_squared.get(dev_int, 0.0)), 0.0)
                observed_exposure = max(float(exposure_by_age.get(dev_int, 0.0)), 1e-12)
                shared_variance += sigma2 / (factor**2 * observed_exposure)
            covariance += 2.0 * float(row_i["Mack Ultimate"]) * float(row_j["Mack Ultimate"]) * shared_variance

    return float(np.sqrt(max(individual_variance + covariance, 0.0)))


def _diagnostics(result: pd.DataFrame, aggregate_standard_error: float) -> dict[str, float]:
    if result.empty:
        total_reserve = total_ultimate = 0.0
    else:
        total_reserve = float(pd.to_numeric(result["Mack Reserve"], errors="coerce").sum())
        total_ultimate = float(pd.to_numeric(result["Mack Ultimate"], errors="coerce").sum())

    total_cv = aggregate_standard_error / total_reserve if total_reserve > 0 else 0.0
    lower75, upper75 = _reserve_interval(total_reserve, aggregate_standard_error, 1.150349)
    lower95, upper95 = _reserve_interval(total_reserve, aggregate_standard_error, 1.959964)
    return {
        "total_mack_reserve": total_reserve,
        "total_mack_ultimate": total_ultimate,
        "total_mack_standard_error": float(aggregate_standard_error),
        "total_mack_cv": float(total_cv),
        "mack_75_lower": float(lower75),
        "mack_75_upper": float(upper75),
        "mack_95_lower": float(lower95),
        "mack_95_upper": float(upper95),
    }
