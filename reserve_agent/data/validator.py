from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    level: str
    message: str


def _numeric_triangle(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _get_ay_col(triangle: pd.DataFrame) -> str:
    mapping = triangle.attrs.get("mapping")

    if mapping is not None and hasattr(mapping, "accident_year_col"):
        return mapping.accident_year_col

    return "事故年"


def _format_excel_rows(value) -> str:
    if isinstance(value, list):
        return "、".join(str(x) for x in value)

    if isinstance(value, tuple):
        return "、".join(str(x) for x in value)

    return str(value)


def _format_records_preview(
    df: pd.DataFrame,
    formatter: Callable,
    n: int = 5,
) -> str:
    if df is None or df.empty:
        return ""

    rows = []

    for i, (_, r) in enumerate(df.head(n).iterrows(), start=1):
        rows.append(f"{i}. {formatter(r)}")

    more = len(df) - n

    if more > 0:
        rows.append(f"另有 {more} 条未展示")

    return "；".join(rows)


def _append_issue(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
    level: str,
    message: str,
) -> None:
    issues.append(ValidationIssue(level, message))

    if level == "error":
        triangle.attrs["model_blocking_error"] = True


def _check_text_garbage(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    garbage_rows = triangle.attrs.get("garbage_rows")

    if not isinstance(garbage_rows, pd.DataFrame) or garbage_rows.empty:
        return

    ay_col = _get_ay_col(triangle)

    preview_text = _format_records_preview(
        garbage_rows,
        lambda r: (
            f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
            f"事故年{r.get(ay_col, r.get('事故年', ''))}，"
            f"列{r.get('DevYear', '')}，"
            f"内容为'{r.get('RawAmount', '')}'"
        ),
    )

    _append_issue(
        triangle,
        issues,
        "error",
        f"所选金额区域共发现 {len(garbage_rows)} 处无法转为数值的数据，"
        f"请重新核查数据。具体为：{preview_text}",
    )


def _check_empty_rows(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    raw = triangle.attrs.get("raw_triangle", triangle)
    num = _numeric_triangle(raw)

    empty_rows = num.index[num.isna().all(axis=1)].tolist()

    if not empty_rows:
        return

    ay_to_rows = triangle.attrs.get("ay_to_rows", {})

    records = pd.DataFrame(
        [
            {
                "Excel行": ay_to_rows.get(ay, ""),
                "事故年": ay,
            }
            for ay in empty_rows
        ]
    )

    preview_text = _format_records_preview(
        records,
        lambda r: (
            f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
            f"事故年{r.get('事故年', '')}"
        ),
    )

    _append_issue(
        triangle,
        issues,
        "error",
        f"共发现 {len(empty_rows)} 个事故年没有任何有效赔款数据，"
        f"请重新核查数据。具体为：{preview_text}",
    )


def _check_triangle_gaps(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    raw = triangle.attrs.get("raw_triangle", triangle)
    num = _numeric_triangle(raw)
    ay_to_rows = triangle.attrs.get("ay_to_rows", {})

    gap_records = []

    for ay, row in num.iterrows():
        valid_positions = np.where(row.notna().to_numpy())[0]

        if len(valid_positions) <= 1:
            continue

        first_pos = valid_positions.min()
        last_pos = valid_positions.max()
        inside = row.iloc[first_pos : last_pos + 1]

        if inside.isna().any():
            missing_cols = inside.index[inside.isna()].tolist()

            gap_records.append(
                {
                    "Excel行": ay_to_rows.get(ay, ""),
                    "事故年": ay,
                    "缺失发展期": missing_cols,
                }
            )

    if not gap_records:
        return

    gap_df = pd.DataFrame(gap_records)

    preview_text = _format_records_preview(
        gap_df,
        lambda r: (
            f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
            f"事故年{r.get('事故年', '')}，"
            f"缺失发展期{r.get('缺失发展期', '')}"
        ),
    )

    _append_issue(
        triangle,
        issues,
        "warning",
        f"共发现 {len(gap_df)} 个事故年存在观测区内部空值，"
        f"系统不会自动填补，将在发展因子估计中剔除受影响的相邻因子。"
        f"具体为：{preview_text}。",
    )


def _check_gap_exclusions(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    gap_exclusions = triangle.attrs.get("gap_exclusions")

    if not isinstance(gap_exclusions, pd.DataFrame) or gap_exclusions.empty:
        return

    preview_text = _format_records_preview(
        gap_exclusions,
        lambda r: (
            f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
            f"事故年{r.get('事故年', '')}，"
            f"缺失发展期{r.get('缺失发展期', '')}，"
            f"剔除因子{r.get('剔除因子', '')}"
        ),
    )

    _append_issue(
        triangle,
        issues,
        "warning",
        f"内部空值共影响 {len(gap_exclusions)} 个相邻发展因子样本，"
        f"系统采用剔除方法处理。具体为：{preview_text}。",
    )


def _check_negative_values(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    is_cumulative = triangle.attrs.get("is_cumulative", True)
    negative_rows = triangle.attrs.get("negative_rows")
    smoothing_log = triangle.attrs.get("smoothing_log")
    ay_col = _get_ay_col(triangle)

    if isinstance(negative_rows, pd.DataFrame) and not negative_rows.empty:
        preview_text = _format_records_preview(
            negative_rows,
            lambda r: (
                f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
                f"事故年{r.get(ay_col, r.get('事故年', ''))}，"
                f"列{r.get('DevYear', '')}，"
                f"金额={r.get('Amount', r.get('RawAmount', ''))}"
            ),
        )

        if is_cumulative:
            _append_issue(
                triangle,
                issues,
                "error",
                f"共发现 {len(negative_rows)} 处累计金额为负值。"
                f"累计赔款口径下负值不符合建模要求，系统已在建模三角中截断为0，"
                f"但仍建议先复核原始数据。具体为：{preview_text}。",
            )
        else:
            _append_issue(
                triangle,
                issues,
                "warning",
                f"共发现 {len(negative_rows)} 处增量金额为负值。"
                f"增量负值可能来自追偿、冲回或数据修正，系统保留原始值。"
                f"具体为：{preview_text}。",
            )

    if isinstance(smoothing_log, pd.DataFrame) and not smoothing_log.empty:
        preview_text = _format_records_preview(
            smoothing_log,
            lambda r: (
                f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
                f"事故年{r.get('事故年', '')}，"
                f"发展期{r.get('发展期', '')}，"
                f"原始累计值={r.get('原始值', '')}，"
                f"修正后={r.get('修正后', '')}"
            ),
        )

        _append_issue(
            triangle,
            issues,
            "error",
            f"累计赔付三角共发现 {len(smoothing_log)} 处负值，"
            f"系统已在建模三角中截断为0。具体为：{preview_text}。",
        )


def _check_cumulative_drop(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    cumulative_drop_log = triangle.attrs.get("cumulative_drop_log")

    if not isinstance(cumulative_drop_log, pd.DataFrame) or cumulative_drop_log.empty:
        return

    preview_text = _format_records_preview(
        cumulative_drop_log,
        lambda r: (
            f"Excel第{_format_excel_rows(r.get('Excel行', ''))}行，"
            f"事故年{r.get('事故年', '')}，"
            f"{r.get('前一发展期', '')}期到{r.get('当前发展期', '')}期，"
            f"累计值由{r.get('前一累计值', '')}下降为{r.get('当前累计值', '')}"
        ),
    )

    _append_issue(
        triangle,
        issues,
        "warning",
        f"累计赔款共发现 {len(cumulative_drop_log)} 处下降。"
        f"系统仅提示，不自动平滑或修改该类下降。具体为：{preview_text}。",
    )


def _check_numeric_structure(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    num = _numeric_triangle(triangle)

    if num.isna().all(axis=None):
        _append_issue(
            triangle,
            issues,
            "error",
            "赔付三角中没有任何可用数值，无法建模。",
        )
        return

    if num.index.duplicated().any():
        duplicated = num.index[num.index.duplicated()].tolist()

        _append_issue(
            triangle,
            issues,
            "error",
            f"赔付三角存在重复事故年：{duplicated}。请检查事故年列或聚合逻辑。",
        )


def _check_model_ready(
    triangle: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    num = _numeric_triangle(triangle)

    n_ay = num.shape[0]
    n_dev = num.shape[1]

    if n_ay == 0 or n_dev == 0:
        _append_issue(
            triangle,
            issues,
            "error",
            "赔付三角没有有效事故年或发展期，模型不可用。",
        )
        return

    if n_dev < 2:
        _append_issue(
            triangle,
            issues,
            "error",
            f"发展期数量仅 {n_dev} 个，无法估计发展因子。",
        )
        return

    if n_ay < 3:
        _append_issue(
            triangle,
            issues,
            "warning",
            f"事故年数量仅 {n_ay} 个，Chain Ladder 稳定性较弱。",
        )

    valid_factor_pairs = {}

    for pos in range(n_dev - 1):
        dev_from = num.columns[pos]
        dev_to = num.columns[pos + 1]

        pair_df = num.iloc[:, [pos, pos + 1]].dropna()
        pair_df = pair_df[pair_df.iloc[:, 0] > 0]

        valid_factor_pairs[f"{dev_from}->{dev_to}"] = len(pair_df)

    no_factor_pairs = [
        pair
        for pair, count in valid_factor_pairs.items()
        if count == 0
    ]

    if valid_factor_pairs and len(no_factor_pairs) == len(valid_factor_pairs):
        _append_issue(
            triangle,
            issues,
            "error",
            "所有相邻发展期都没有可用于估计发展因子的有效数据，已阻断模型运行。",
        )

    elif no_factor_pairs:
        _append_issue(
            triangle,
            issues,
            "warning",
            f"以下发展期没有有效因子样本，将无法稳定估计：{no_factor_pairs}",
        )

    triangle.attrs["valid_factor_pairs"] = valid_factor_pairs

    observed_count = int(num.notna().sum().sum())

    if observed_count < 6:
        _append_issue(
            triangle,
            issues,
            "warning",
            f"有效观测点仅 {observed_count} 个，准备金估计可靠性较低。",
        )


def validate_triangle(triangle: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if triangle is None or triangle.empty:
        return [ValidationIssue("error", "赔付三角为空，无法建模。")]

    triangle.attrs["model_blocking_error"] = False

    _check_numeric_structure(triangle, issues)
    _check_text_garbage(triangle, issues)
    _check_empty_rows(triangle, issues)
    _check_triangle_gaps(triangle, issues)
    _check_gap_exclusions(triangle, issues)
    _check_negative_values(triangle, issues)
    _check_cumulative_drop(triangle, issues)
    _check_model_ready(triangle, issues)

    if any(issue.level == "error" for issue in issues):
        triangle.attrs["model_blocking_error"] = True

    if not issues:
        issues.append(
            ValidationIssue(
                "info",
                "数据状态良好，未发现明显结构、类型或业务质量问题。",
            )
        )

    return issues