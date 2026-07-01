from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reserve_agent.models.mack import mack_chain_ladder
from reserve_agent.models.reserving_core import (
    age_to_ultimate_factors,
    bornhuetter_ferguson,
    chain_ladder,
    expected_loss_ratio,
    volume_weighted_factors,
)
from reserve_agent.models.sensitivity import expected_loss_ratio_sensitivity, factor_sensitivity


@dataclass
class ReservingOutputs:
    selected_factors: pd.Series
    age_to_ultimate: pd.Series
    chain_ladder: pd.DataFrame
    expected_loss_ratio: pd.DataFrame
    bornhuetter_ferguson: pd.DataFrame
    comparison: pd.DataFrame
    diagnostics: dict[str, float]
    mack: pd.DataFrame | None = None
    mack_diagnostics: dict[str, float] | None = None
    expected_lr_sensitivity: pd.DataFrame | None = None
    factor_sensitivity: pd.DataFrame | None = None


def compare_methods(
    cl: pd.DataFrame,
    elr: pd.DataFrame,
    bf: pd.DataFrame,
    mack: pd.DataFrame | None = None,
) -> pd.DataFrame:
    comparison = cl[["Accident Year", "Latest Cumulative", "Ultimate Loss", "Reserve"]].rename(
        columns={"Ultimate Loss": "Chain Ladder Ultimate", "Reserve": "Chain Ladder Reserve"}
    )
    comparison = comparison.merge(
        elr[["Accident Year", "Expected Ultimate Loss", "Reserve"]].rename(columns={"Reserve": "ELR Reserve"}),
        on="Accident Year",
        how="left",
    )
    comparison = comparison.merge(
        bf[["Accident Year", "BF Ultimate Loss", "BF Reserve"]],
        on="Accident Year",
        how="left",
    )
    if mack is not None and not mack.empty:
        comparison = comparison.merge(
            mack[
                [
                    "Accident Year",
                    "Mack Ultimate",
                    "Mack Reserve",
                    "Mack Standard Error",
                    "Mack CV",
                    "Mack 95% Lower",
                    "Mack 95% Upper",
                ]
            ],
            on="Accident Year",
            how="left",
        )
    return comparison


def run_reserving_models(
    cumulative_triangle: pd.DataFrame,
    exposure: pd.DataFrame | None = None,
    expected_loss_ratio_value: float = 0.72,
) -> ReservingOutputs:
    factors, atu, cl = chain_ladder(cumulative_triangle)
    elr = expected_loss_ratio(cumulative_triangle, exposure, expected_loss_ratio_value)
    bf = bornhuetter_ferguson(cumulative_triangle, atu, elr)
    mack_result = mack_chain_ladder(cumulative_triangle, factors, atu)
    comparison = compare_methods(cl, elr, bf, mack_result.by_year)

    expected_lr_sens = expected_loss_ratio_sensitivity(
        cumulative_triangle,
        atu,
        exposure,
        base_expected_lr=expected_loss_ratio_value,
    )
    factor_sens = factor_sensitivity(cumulative_triangle, factors)

    diagnostics = {
        "total_latest": float(comparison["Latest Cumulative"].sum()),
        "total_cl_reserve": float(comparison["Chain Ladder Reserve"].sum()),
        "total_elr_reserve": float(comparison["ELR Reserve"].sum()),
        "total_bf_reserve": float(comparison["BF Reserve"].sum()),
    }
    diagnostics.update(mack_result.diagnostics)

    return ReservingOutputs(
        selected_factors=factors,
        age_to_ultimate=atu,
        chain_ladder=cl,
        expected_loss_ratio=elr,
        bornhuetter_ferguson=bf,
        comparison=comparison,
        diagnostics=diagnostics,
        mack=mack_result.by_year,
        mack_diagnostics=mack_result.diagnostics,
        expected_lr_sensitivity=expected_lr_sens,
        factor_sensitivity=factor_sens,
    )


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}m"
    if abs_value >= 1_000:
        return f"{value / 1_000:,.1f}k"
    return f"{value:,.0f}"
