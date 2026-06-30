from __future__ import annotations

import numpy as np
import pandas as pd

from reserve_agent.models.reserving_core import (
    age_to_ultimate_factors,
    bornhuetter_ferguson,
    expected_loss_ratio,
    project_chain_ladder_with_factors,
)


def expected_loss_ratio_sensitivity(
    cumulative_triangle: pd.DataFrame,
    age_to_ultimate: pd.Series,
    exposure: pd.DataFrame | None = None,
    base_expected_lr: float = 0.72,
    low: float = 0.50,
    high: float = 1.00,
    step: float = 0.05,
) -> pd.DataFrame:
    """Measure how ELR and BF reserves change under expected LR assumptions."""

    grid = sorted({round(float(value), 4) for value in np.arange(low, high + step / 2, step)} | {round(base_expected_lr, 4)})
    rows: list[dict[str, float | str]] = []
    for lr in grid:
        elr = expected_loss_ratio(cumulative_triangle, exposure, lr)
        bf = bornhuetter_ferguson(cumulative_triangle, age_to_ultimate, elr)
        rows.append(
            {
                "Scenario": "Base" if abs(lr - base_expected_lr) < 1e-9 else f"ELR {lr:.0%}",
                "Expected Loss Ratio": lr,
                "ELR Reserve": float(elr["Reserve"].sum()),
                "ELR Ultimate": float(elr["Expected Ultimate Loss"].sum()),
                "BF Reserve": float(bf["BF Reserve"].sum()),
                "BF Ultimate": float(bf["BF Ultimate Loss"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Expected Loss Ratio").reset_index(drop=True)


def factor_sensitivity(
    cumulative_triangle: pd.DataFrame,
    selected_factors: pd.Series,
    shocks: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
) -> pd.DataFrame:
    """Measure chain ladder reserve sensitivity to selected factor shocks."""

    rows: list[dict[str, float | str]] = []
    for shock in shocks:
        shocked_factors = selected_factors.astype(float) * (1.0 + shock)
        shocked_factors = shocked_factors.clip(lower=1.0)
        shocked_atu = age_to_ultimate_factors(shocked_factors, list(cumulative_triangle.columns))
        projected = project_chain_ladder_with_factors(cumulative_triangle, shocked_factors, shocked_atu)
        rows.append(
            {
                "Scenario": "Base" if abs(shock) < 1e-12 else f"{shock:+.0%} factor shock",
                "Factor Shock": shock,
                "Chain Ladder Reserve": float(projected["Reserve"].sum()),
                "Chain Ladder Ultimate": float(projected["Ultimate Loss"].sum()),
                "Average Selected Factor": float(shocked_factors.mean()) if len(shocked_factors) else 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values("Factor Shock").reset_index(drop=True)
