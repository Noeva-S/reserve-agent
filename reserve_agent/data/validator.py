from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    level: str
    message: str


def validate_triangle(triangle: pd.DataFrame) -> list[ValidationIssue]:
    """Run practical checks on a cumulative loss triangle."""

    issues: list[ValidationIssue] = []
    if triangle is None or triangle.empty:
        return [ValidationIssue("error", "赔付三角为空，无法建模。")]

    numeric = triangle.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        return [ValidationIssue("error", "三角形中没有可用的数值。")]

    if numeric.index.duplicated().any():
        issues.append(ValidationIssue("error", "三角形存在重复事故年。"))

    negative_count = int((numeric < 0).sum().sum())
    if negative_count:
        issues.append(ValidationIssue("warning", f"三角形中存在 {negative_count} 个负值，请复核追偿、冲回或录入修正。"))

    for ay, row in numeric.iterrows():
        observed_positions = np.where(row.notna().to_numpy())[0]
        if len(observed_positions) <= 1:
            continue
        first_pos = observed_positions.min()
        last_pos = observed_positions.max()
        inner = row.iloc[first_pos : last_pos + 1]
        if inner.isna().any():
            missing_cols = [str(col) for col in inner.index[inner.isna()].tolist()]
            issues.append(ValidationIssue("warning", f"事故年 {ay} 在观察区间内缺少发展期 {', '.join(missing_cols)}。"))

    for ay, row in numeric.iterrows():
        observed = row.dropna()
        if len(observed) <= 1:
            continue
        drops = observed.diff().dropna()
        if (drops < 0).any():
            issues.append(ValidationIssue("warning", f"事故年 {ay} 的累计赔款出现下降，请检查是否存在冲回或口径变化。"))

    if numeric.shape[1] < 2:
        issues.append(ValidationIssue("error", "发展期少于 2 列，无法估计发展因子。"))
    if numeric.shape[0] < 3:
        issues.append(ValidationIssue("warning", "事故年数量少于 3 个，模型稳定性较弱。"))

    valid_factor_pairs = {}
    for pos in range(numeric.shape[1] - 1):
        dev_from = numeric.columns[pos]
        dev_to = numeric.columns[pos + 1]
        pair = numeric.iloc[:, [pos, pos + 1]].dropna()
        pair = pair[pair.iloc[:, 0] > 0]
        valid_factor_pairs[f"{dev_from}->{dev_to}"] = len(pair)
    no_factor_pairs = [pair for pair, count in valid_factor_pairs.items() if count == 0]
    if valid_factor_pairs and len(no_factor_pairs) == len(valid_factor_pairs):
        issues.append(ValidationIssue("error", "所有相邻发展期都缺少可用于估计发展因子的有效样本。"))
    elif no_factor_pairs:
        issues.append(ValidationIssue("warning", f"以下发展期没有有效因子样本：{', '.join(no_factor_pairs)}。"))

    if not issues:
        issues.append(ValidationIssue("info", "数据结构检查通过，未发现明显阻断建模的问题。"))
    return issues
