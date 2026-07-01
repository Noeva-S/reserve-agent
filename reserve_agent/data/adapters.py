from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import numpy as np
import pandas as pd

from reserve_agent.data.detector import (
    ExcelFormat,
    find_role_column,
    normalise_label,
    parse_development_label,
)
from reserve_agent.data.loader import build_cumulative_triangle
from reserve_agent.data.mapping import FieldMapping

EXCEL_ROW_COL = "__excel_row__"


def _coerce_integer_axis(series: pd.Series, axis_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        raise ValueError(f"{axis_name}列没有可识别的数字。")
    return numeric


def _find_preferred_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Any | None:
    columns_by_label = {normalise_label(column): column for column in df.columns}
    for alias in aliases:
        if alias in columns_by_label:
            return columns_by_label[alias]
    return None


def _is_claim_development_table(df: pd.DataFrame) -> bool:
    claim_col = find_role_column(df, "claim_id")
    labels = {normalise_label(column) for column in df.columns}
    return claim_col is not None and bool(
        labels
        & {
            "delayyears",
            "delayyear",
            "delaythirds",
            "delayshifted",
            "delaydays",
            "delayday",
            "delaymonths",
            "delaymonth",
            "reportingdelay",
            "reportingdelaydays",
        }
    )


def _normalise_development_values(series: pd.Series, column: Any) -> pd.Series:
    """Convert explicit day/month delays to annual development buckets."""

    numeric = pd.to_numeric(series, errors="coerce")
    label = normalise_label(column)
    if "day" in label or "天" in label:
        return numeric // 365.25
    if "month" in label or "月" in label:
        return numeric // 12
    return numeric


def _find_amount_column(df: pd.DataFrame, measure: str) -> Any | None:
    if measure.strip().lower() == "incurred":
        priorities = (
            "overallamountrevalued",
            "totalincurred",
            "incurred",
            "overallamountdestinationcurrency",
            "overallamount",
            "amount",
            "value",
            "loss",
            "金额",
            "赔款金额",
        )
    else:
        priorities = ("paid", "paidamount", "amount", "value", "loss", "金额", "赔款金额")
    preferred = _find_preferred_column(df, priorities)
    return preferred if preferred is not None else find_role_column(df, "amount")


def _incremental_to_cumulative(triangle: pd.DataFrame) -> pd.DataFrame:
    if triangle.empty:
        return triangle

    latest_calendar_period = max(
        int(accident_year) + int(development)
        for accident_year, row in triangle.iterrows()
        for development, value in row.items()
        if pd.notna(value)
    )
    max_development = max(int(max(triangle.columns)), latest_calendar_period - int(min(triangle.index)))
    all_developments = list(range(0, max_development + 1))
    incremental = triangle.reindex(columns=all_developments)

    cumulative = pd.DataFrame(index=incremental.index, columns=all_developments, dtype=float)
    for accident_year, row in incremental.iterrows():
        latest_observable_development = min(max_development, latest_calendar_period - int(accident_year))
        observed_columns = [
            development for development in all_developments if development <= latest_observable_development
        ]
        if observed_columns:
            cumulative.loc[accident_year, observed_columns] = row[observed_columns].fillna(0.0).cumsum().to_numpy()
    cumulative.index.name = "Accident Year"
    return cumulative


def claims_snapshot_to_triangle(df: pd.DataFrame, measure: str = "Paid") -> pd.DataFrame:
    claim_col = find_role_column(df, "claim_id")
    accident_col = find_role_column(df, "accident_year")
    measure_col = find_role_column(df, "measure")
    if claim_col is None or accident_col is None or measure_col is None:
        raise ValueError("赔案明细格式需要 Claim ID、Loss Year 和 Type 字段。")

    canonical = df.rename(
        columns={
            claim_col: "Claim ID",
            accident_col: "Loss Year",
            measure_col: "Type",
        }
    ).copy()
    return build_cumulative_triangle(canonical, measure=measure)


def triangle_sheet_to_triangle(df: pd.DataFrame) -> pd.DataFrame:
    accident_col = find_role_column(df, "accident_year")
    if accident_col is not None:
        accident_year = _coerce_integer_axis(df[accident_col], "事故年")
        value_df = df.drop(columns=[accident_col])
    elif df.index.name is not None:
        accident_year = _coerce_integer_axis(pd.Series(df.index, index=df.index), "事故年")
        value_df = df.copy()
    else:
        raise ValueError("三角形格式缺少事故年列。")

    columns_by_development: dict[int, list[Any]] = defaultdict(list)
    for column in value_df.columns:
        development = parse_development_label(column)
        if development is not None:
            columns_by_development[development].append(column)
    if not columns_by_development:
        raise ValueError("三角形格式没有可识别的发展期列。")

    valid_rows = accident_year.notna()
    data: dict[int, pd.Series] = {}
    for development, source_columns in columns_by_development.items():
        numeric = value_df.loc[valid_rows, source_columns].apply(pd.to_numeric, errors="coerce")
        data[development] = numeric.sum(axis=1, min_count=1)

    triangle = pd.DataFrame(data)
    triangle.index = accident_year.loc[valid_rows].astype(int).to_numpy()
    triangle.index.name = "Accident Year"
    triangle = triangle.groupby(level=0).sum(min_count=1)
    triangle = triangle.sort_index().sort_index(axis=1).dropna(axis=1, how="all")
    if triangle.empty:
        raise ValueError("三角形数据区域为空。")
    return triangle.astype(float)


def long_table_to_triangle(
    df: pd.DataFrame,
    *,
    is_cumulative: bool = True,
    measure: str = "Paid",
) -> pd.DataFrame:
    claim_development_table = _is_claim_development_table(df)
    if claim_development_table:
        accident_col = _find_preferred_column(
            df,
            ("policyyearshifted", "lossyear", "accidentyear", "originyear", "policyyear"),
        )
        development_col = _find_preferred_column(
            df,
            (
                "delayshifted",
                "delayyears",
                "delayyear",
                "delaymonths",
                "delaymonth",
                "delaydays",
                "delayday",
                "reportingdelaydays",
                "reportingdelay",
                "delaythirds",
            ),
        )
    else:
        accident_col = find_role_column(df, "accident_year")
        development_col = find_role_column(df, "development")
    amount_col = _find_amount_column(df, measure)
    if accident_col is None or development_col is None or amount_col is None:
        raise ValueError("长表格式需要事故年、发展期和金额字段。")

    long_df = df[[accident_col, development_col, amount_col]].copy()
    long_df.columns = ["accident_year", "development", "amount"]
    long_df["accident_year"] = pd.to_numeric(long_df["accident_year"], errors="coerce")
    long_df["development"] = _normalise_development_values(
        long_df["development"], development_col
    )
    long_df["amount"] = pd.to_numeric(long_df["amount"], errors="coerce")
    long_df = long_df.dropna(subset=["accident_year", "development", "amount"])
    long_df = long_df[long_df["development"] >= 0]
    if long_df.empty:
        raise ValueError("长表中没有有效的事故年、发展期和金额记录。")

    long_df["accident_year"] = long_df["accident_year"].astype(int)
    long_df["development"] = long_df["development"].astype(int)
    triangle = long_df.pivot_table(
        index="accident_year",
        columns="development",
        values="amount",
        aggfunc="sum",
    ).sort_index().sort_index(axis=1)

    if claim_development_table or not is_cumulative:
        triangle = _incremental_to_cumulative(triangle)
    triangle.index.name = "Accident Year"
    return triangle.astype(float)


def adapt_to_triangle(
    df: pd.DataFrame,
    format_name: ExcelFormat,
    *,
    measure: str = "Paid",
    is_cumulative: bool = True,
) -> pd.DataFrame:
    if format_name == "claims_snapshot":
        return claims_snapshot_to_triangle(df, measure=measure)
    if format_name == "triangle":
        return triangle_sheet_to_triangle(df)
    if format_name == "long_table":
        return long_table_to_triangle(df, is_cumulative=is_cumulative, measure=measure)
    raise ValueError("无法识别 Excel 格式，不能转换为准备金三角形。")


# ---------------------------------------------------------------------------
# 兼容队友第 3 部分：字段映射确认后的标准三角转换
# ---------------------------------------------------------------------------

def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def _normalize_blank_to_na(s: pd.Series) -> pd.Series:
    return s.replace(
        {
            "": np.nan,
            " ": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "NULL": np.nan,
            "-": np.nan,
            "--": np.nan,
        }
    )


def _to_numeric_amount(s: pd.Series) -> pd.Series:
    """
    将金额列尽量转成数值。

    支持：
    - 普通数字；
    - 带千分位逗号的数字；
    - 带常见货币符号的数字；
    - 空字符串、"-"、"--" 等视为空值。
    """
    raw = _normalize_blank_to_na(s)

    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_numeric(raw, errors="coerce")

    text = raw.astype("string").str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"[￥¥$£€]", "", regex=True)
    text = text.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
        }
    )

    return pd.to_numeric(text, errors="coerce")


def _extract_first_number(s: pd.Series) -> pd.Series:
    """
    从发展期或日历年标签中提取数字。

    例如：
    - 0, 1, 2
    - "Dev 1"
    - "Development 2"
    - "2018年"
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    text = s.astype("string").str.strip()
    text = text.str.replace("年", "", regex=False)
    text = text.str.replace("期", "", regex=False)

    extracted = text.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce")

    return numeric


def _infer_dev_age(dev_year_series: pd.Series, accident_year_series: pd.Series) -> pd.Series:
    """
    识别发展期。

    支持两种常见情况：
    1. 金额列名或发展期列本身就是发展期：0, 1, 2, 3...
    2. 金额列名或发展期列是日历年：2007, 2008, 2009...
       此时发展期 = 日历年 - 事故年
    """
    dev_num = _extract_first_number(dev_year_series)
    ay_num = pd.to_numeric(accident_year_series, errors="coerce")

    non_na_dev = dev_num.dropna()

    if not non_na_dev.empty and non_na_dev.max() <= 60:
        return dev_num

    return dev_num - ay_num


def _sum_with_nan(series: pd.Series):
    return series.sum(min_count=1)


def _clip_negative_cumulative_only(triangle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    只处理累计负值。

    规则：
    - 累计负值不符合累计赔款建模口径；
    - 建模三角中将负值截断为 0；
    - 不处理累计下降，累计下降只提示。
    """
    num = triangle.apply(pd.to_numeric, errors="coerce")
    clipped = num.copy()

    records = []

    for ay in clipped.index:
        for dev in clipped.columns:
            val = clipped.loc[ay, dev]

            if pd.notna(val) and val < 0:
                records.append(
                    {
                        "事故年": ay,
                        "发展期": dev,
                        "原始值": val,
                        "修正后": 0,
                        "原因": "累计负值截断为0",
                    }
                )
                clipped.loc[ay, dev] = 0

    clipped.index.name = "Accident Year"

    return clipped, pd.DataFrame(records)


def _build_gap_exclusions(triangle: pd.DataFrame, ay_to_rows: dict) -> pd.DataFrame:
    """
    内部空值不填补。

    如果某事故年在观察区间内部存在空值，则涉及该空值的相邻发展因子样本应当剔除。
    """
    records = []
    num = triangle.apply(pd.to_numeric, errors="coerce")
    cols = list(num.columns)

    for ay, row in num.iterrows():
        valid_positions = np.where(row.notna().to_numpy())[0]

        if len(valid_positions) <= 1:
            continue

        first_pos = valid_positions.min()
        last_pos = valid_positions.max()

        for pos in range(first_pos, last_pos + 1):
            if pd.isna(row.iloc[pos]):
                dev = cols[pos]
                excel_rows = ay_to_rows.get(ay, [])

                if pos - 1 >= 0:
                    records.append(
                        {
                            "Excel行": excel_rows,
                            "事故年": ay,
                            "缺失发展期": dev,
                            "剔除因子": f"{cols[pos - 1]}->{dev}",
                            "原因": "缺失单元作为发展因子分子",
                        }
                    )

                if pos + 1 < len(cols):
                    records.append(
                        {
                            "Excel行": excel_rows,
                            "事故年": ay,
                            "缺失发展期": dev,
                            "剔除因子": f"{dev}->{cols[pos + 1]}",
                            "原因": "缺失单元作为发展因子分母",
                        }
                    )

    return pd.DataFrame(records)


def _build_cumulative_drop_log(triangle: pd.DataFrame, ay_to_rows: dict) -> pd.DataFrame:
    """
    累计下降只提示，不自动平滑。
    """
    records = []
    num = triangle.apply(pd.to_numeric, errors="coerce")

    for ay, row in num.iterrows():
        observed = row.dropna()

        if len(observed) <= 1:
            continue

        prev_dev = None
        prev_val = None

        for dev, val in observed.items():
            if prev_val is not None and val < prev_val:
                records.append(
                    {
                        "Excel行": ay_to_rows.get(ay, []),
                        "事故年": ay,
                        "前一发展期": prev_dev,
                        "当前发展期": dev,
                        "前一累计值": prev_val,
                        "当前累计值": val,
                        "原因": "累计下降，仅提示，不自动修改",
                    }
                )

            prev_dev = dev
            prev_val = val

    return pd.DataFrame(records)


def _build_observed_mask(
    long_df: pd.DataFrame,
    ay_col: str,
    model_triangle: pd.DataFrame,
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            False,
            index=model_triangle.index,
            columns=model_triangle.columns,
        )

    temp = long_df.copy()
    temp["_observed"] = _normalize_blank_to_na(temp["RawAmount"]).notna()

    observed_mask = (
        temp.pivot_table(
            values="_observed",
            index=ay_col,
            columns="DevAge",
            aggfunc="max",
            fill_value=False,
        )
        .reindex(
            index=model_triangle.index,
            columns=model_triangle.columns,
            fill_value=False,
        )
        .astype(bool)
    )

    return observed_mask


def _finalize_triangle_from_long_records(
    long_df: pd.DataFrame,
    *,
    ay_col: str,
    processed_df: pd.DataFrame,
    mapping: FieldMapping,
    ay_to_rows: dict,
    data_format: str,
) -> pd.DataFrame:
    """
    将统一后的长格式记录转换为标准累计赔付三角。

    long_df 必须包含：
    - EXCEL_ROW_COL
    - ay_col
    - DevYear
    - RawAmount
    """
    raw_amount = _normalize_blank_to_na(long_df["RawAmount"])
    amount_num = _to_numeric_amount(raw_amount)

    garbage_mask = raw_amount.notna() & amount_num.isna()
    garbage_rows = long_df.loc[
        garbage_mask,
        [EXCEL_ROW_COL, ay_col, "DevYear", "RawAmount"],
    ].copy()
    garbage_rows = garbage_rows.rename(columns={EXCEL_ROW_COL: "Excel行"})

    negative_mask = amount_num < 0
    negative_rows = long_df.loc[
        negative_mask,
        [EXCEL_ROW_COL, ay_col, "DevYear", "RawAmount"],
    ].copy()
    negative_rows["Amount"] = amount_num.loc[negative_mask].values
    negative_rows = negative_rows.rename(columns={EXCEL_ROW_COL: "Excel行"})

    long_df = long_df.copy()
    long_df["Amount"] = amount_num
    long_df["DevAge"] = _infer_dev_age(long_df["DevYear"], long_df[ay_col])

    long_df = long_df.dropna(subset=["DevAge"]).copy()
    long_df["DevAge"] = long_df["DevAge"].astype(int)
    long_df = long_df[long_df["DevAge"] >= 0].copy()

    if long_df.empty:
        raise ValueError("无法识别任何有效发展期，请检查金额列表头或发展期列是否为发展期或日历年。")

    grouped = (
        long_df.groupby([ay_col, "DevAge"], dropna=False)["Amount"]
        .apply(_sum_with_nan)
    )

    raw_input_triangle = grouped.unstack("DevAge")
    raw_input_triangle = raw_input_triangle.sort_index().sort_index(axis=1)

    if raw_input_triangle.empty or raw_input_triangle.shape[1] == 0:
        raise ValueError("生成的赔付三角为空，请检查事故年列、发展期列和金额列映射。")

    if mapping.is_cumulative:
        raw_cumulative_triangle = raw_input_triangle.copy()
    else:
        raw_cumulative_triangle = raw_input_triangle.cumsum(axis=1)

    raw_cumulative_triangle.index.name = "Accident Year"

    gap_exclusions = _build_gap_exclusions(raw_cumulative_triangle, ay_to_rows)
    cumulative_drop_log = _build_cumulative_drop_log(raw_cumulative_triangle, ay_to_rows)

    model_triangle, negative_cumulative_smoothing_log = _clip_negative_cumulative_only(
        raw_cumulative_triangle
    )

    if not negative_cumulative_smoothing_log.empty:
        negative_cumulative_smoothing_log["Excel行"] = negative_cumulative_smoothing_log["事故年"].map(
            lambda x: ay_to_rows.get(x, [])
        )

    observed_mask = _build_observed_mask(
        long_df=long_df,
        ay_col=ay_col,
        model_triangle=model_triangle,
    )

    model_triangle.attrs.update(
        {
            "processed_data": processed_df,
            "long_data": long_df,
            "raw_input_triangle": raw_input_triangle,
            "raw_triangle": raw_cumulative_triangle,
            "observed_mask": observed_mask,
            "garbage_rows": garbage_rows,
            "negative_rows": negative_rows,
            "gap_exclusions": gap_exclusions,
            "cumulative_drop_log": cumulative_drop_log,
            "smoothing_log": negative_cumulative_smoothing_log,
            "has_text_garbage": not garbage_rows.empty,
            "has_negative_raw_values": not negative_rows.empty,
            "was_smoothed": not negative_cumulative_smoothing_log.empty,
            "is_cumulative": mapping.is_cumulative,
            "mapping": mapping,
            "ay_to_rows": ay_to_rows,
            "data_format": data_format,
        }
    )

    return model_triangle


def _prepare_work_df(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df.columns = [str(c) for c in work_df.columns]

    if EXCEL_ROW_COL not in work_df.columns:
        work_df[EXCEL_ROW_COL] = work_df.index + 2

    return work_df


def _apply_measure_filter(
    work_df: pd.DataFrame,
    mapping: FieldMapping,
) -> pd.DataFrame:
    type_col = str(mapping.type_col_name or "")

    if not type_col:
        return work_df.copy()

    if type_col not in work_df.columns:
        raise ValueError(f"口径列不存在：{type_col}")

    filtered = work_df[
        work_df[type_col].astype(str).str.strip()
        == str(mapping.measure_col_value).strip()
    ].copy()

    if filtered.empty:
        raise ValueError(f"按口径 {mapping.measure_col_value} 过滤后没有可用数据。")

    return filtered


def _build_ay_to_rows(processed_df: pd.DataFrame, ay_col: str) -> dict:
    return (
        processed_df.groupby(ay_col)[EXCEL_ROW_COL]
        .apply(lambda x: sorted(set(pd.to_numeric(x, errors="coerce").dropna().astype(int).tolist())))
        .to_dict()
    )


def _transform_wide_table(
    work_df: pd.DataFrame,
    mapping: FieldMapping,
    *,
    ay_col: str,
    amount_cols: list[str],
) -> pd.DataFrame:
    processed_df = work_df[[EXCEL_ROW_COL, ay_col] + amount_cols].copy()

    processed_df[ay_col] = pd.to_numeric(processed_df[ay_col], errors="coerce")
    processed_df = processed_df.dropna(subset=[ay_col]).copy()

    if processed_df.empty:
        raise ValueError("口径过滤或事故年清洗后没有可用数据。")

    processed_df[ay_col] = processed_df[ay_col].astype(int)

    ay_to_rows = _build_ay_to_rows(processed_df, ay_col)

    long_df = processed_df.melt(
        id_vars=[EXCEL_ROW_COL, ay_col],
        value_vars=amount_cols,
        var_name="DevYear",
        value_name="RawAmount",
    )

    return _finalize_triangle_from_long_records(
        long_df,
        ay_col=ay_col,
        processed_df=processed_df,
        mapping=mapping,
        ay_to_rows=ay_to_rows,
        data_format="wide_table",
    )


def _transform_long_table(
    work_df: pd.DataFrame,
    mapping: FieldMapping,
    *,
    ay_col: str,
    dev_col: str,
    amount_col: str,
) -> pd.DataFrame:
    processed_df = work_df[
        [
            EXCEL_ROW_COL,
            ay_col,
            dev_col,
            amount_col,
        ]
    ].copy()

    processed_df[ay_col] = pd.to_numeric(processed_df[ay_col], errors="coerce")
    processed_df = processed_df.dropna(subset=[ay_col]).copy()

    if processed_df.empty:
        raise ValueError("口径过滤或事故年清洗后没有可用数据。")

    processed_df[ay_col] = processed_df[ay_col].astype(int)

    ay_to_rows = _build_ay_to_rows(processed_df, ay_col)

    long_df = processed_df.rename(
        columns={
            dev_col: "DevYear",
            amount_col: "RawAmount",
        }
    )[
        [
            EXCEL_ROW_COL,
            ay_col,
            "DevYear",
            "RawAmount",
        ]
    ].copy()

    return _finalize_triangle_from_long_records(
        long_df,
        ay_col=ay_col,
        processed_df=processed_df,
        mapping=mapping,
        ay_to_rows=ay_to_rows,
        data_format="long_table",
    )


def transform_to_standard_triangle(df: pd.DataFrame, mapping: FieldMapping) -> pd.DataFrame:
    """
    根据用户手动字段映射，生成标准累计赔付三角。

    支持：
    1. 宽表 / 三角形：
       - 事故年列；
       - 多个金额列；
       - 金额列名可以是发展期 0,1,2...，也可以是日历年 2018,2019...

    2. 长表：
       - 事故年列；
       - 发展期列；
       - 一个金额列；
       - 当 development_col 非空，且 amount_cols 只有一个时，自动按长表处理。

    输出：
    - 返回建模用累计三角；
    - 同时把原始诊断信息写入 triangle.attrs，供 validator.py 使用。
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空，无法生成赔付三角。")

    ay_col = str(mapping.accident_year_col or "").strip()
    dev_col = str(mapping.development_col or "").strip()
    amount_cols = _dedupe_keep_order(
        [
            str(c).strip()
            for c in (mapping.amount_cols or [])
            if str(c).strip()
        ]
    )

    if not ay_col or not amount_cols:
        raise ValueError("请至少指定事故年列和金额列。")

    if ay_col in amount_cols:
        raise ValueError("金额列不能包含事故年列，请重新选择字段映射。")

    work_df = _prepare_work_df(df)

    missing_base_cols = []

    if ay_col not in work_df.columns:
        missing_base_cols.append(ay_col)

    for col in amount_cols:
        if col not in work_df.columns:
            missing_base_cols.append(col)

    if dev_col and dev_col not in work_df.columns:
        missing_base_cols.append(dev_col)

    if missing_base_cols:
        raise ValueError(f"映射列不存在：{missing_base_cols}")

    work_df = _apply_measure_filter(work_df, mapping)

    is_long_table = bool(
        dev_col
        and dev_col in work_df.columns
        and len(amount_cols) == 1
        and dev_col not in amount_cols
    )

    if is_long_table:
        return _transform_long_table(
            work_df,
            mapping,
            ay_col=ay_col,
            dev_col=dev_col,
            amount_col=amount_cols[0],
        )

    return _transform_wide_table(
        work_df,
        mapping,
        ay_col=ay_col,
        amount_cols=amount_cols,
    )