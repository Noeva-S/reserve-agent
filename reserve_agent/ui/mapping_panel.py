from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from reserve_agent.data.mapping import FieldMapping
from reserve_agent.data.table_scanner import (
    get_excel_sheet_shape,
    scan_excel_table,
    scan_excel_table_region,
)


EXCEL_ROW_COL = "__excel_row__"


@dataclass
class MappingPanelResult:
    mapping: FieldMapping
    data_format: str
    selected_measure: str
    header_row: int
    preview_df: pd.DataFrame
    confirmed: bool
    errors: list[str]
    warnings: list[str]


def _get_value(source: dict, *keys: str, default=None):
    for key in keys:
        if key in source and source[key] not in [None, ""]:
            return source[key]
    return default


def _safe_index(options: list[str], value: Any) -> int:
    value = "" if value is None else str(value)
    return options.index(value) if value in options else 0


def _normalise_amount_cols(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _visible_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) != EXCEL_ROW_COL]


def _numeric_non_na_count(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def _validate_mapping(
    preview_df: pd.DataFrame,
    mapping: FieldMapping,
    data_format: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    visible_columns = _visible_columns(preview_df)

    if not mapping.accident_year_col:
        errors.append("缺少事故年列。请选择事故年、出险年度、AY 或 Accident Year 对应字段。")
    elif mapping.accident_year_col not in visible_columns:
        errors.append(f"事故年列 {mapping.accident_year_col} 不存在。")
    elif _numeric_non_na_count(preview_df[mapping.accident_year_col]) == 0:
        errors.append("事故年列无法转成数字，请重新选择事故年字段。")

    if not mapping.amount_cols:
        errors.append("缺少金额列。请至少选择一个赔款金额列。")
    else:
        missing = [c for c in mapping.amount_cols if c not in visible_columns]
        if missing:
            errors.append(f"金额列不存在：{missing}")
        else:
            numeric_counts = [_numeric_non_na_count(preview_df[c]) for c in mapping.amount_cols]
            if sum(numeric_counts) == 0:
                errors.append("金额列无法转成数字，请重新选择金额字段。")
            elif any(count == 0 for count in numeric_counts):
                bad_cols = [c for c, count in zip(mapping.amount_cols, numeric_counts) if count == 0]
                warnings.append(f"以下金额列暂未识别到有效数值：{bad_cols}。")

    if data_format == "long_table":
        if not mapping.development_col:
            errors.append("长表缺少发展期列。请选择 Development、Delay、Lag 或发展期字段。")
        if len(mapping.amount_cols) > 1:
            errors.append("长表金额列只能选择一个。")

    if mapping.development_col and mapping.development_col not in visible_columns:
        errors.append(f"发展期列 {mapping.development_col} 不存在。")

    if mapping.type_col_name and mapping.type_col_name not in visible_columns:
        errors.append(f"口径列 {mapping.type_col_name} 不存在。")

    if mapping.type_col_name and not str(mapping.measure_col_value).strip():
        errors.append("已选择口径列，但目标口径为空。")

    return errors, warnings


def _scan_selected_region(
    file_path: str,
    sheet_name: str,
    region_mode: str,
    default_header_row: int,
    key_prefix: str,
) -> tuple[pd.DataFrame, int, list[str]]:
    errors: list[str] = []

    n_rows, n_cols = get_excel_sheet_shape(file_path, sheet_name)

    if n_rows == 0 or n_cols == 0:
        return pd.DataFrame(), default_header_row, ["当前工作表为空。"]

    if region_mode == "使用整张工作表的有效数据":
        default_header_excel_row = min(max(int(default_header_row) + 1, 1), n_rows)

        header_row_excel = int(
            st.number_input(
                "表头所在行号（从1开始）",
                min_value=1,
                max_value=max(n_rows, 1),
                value=default_header_excel_row,
                step=1,
                key=f"{key_prefix}_whole_header_row",
            )
        )

        header_row = header_row_excel - 1

        try:
            preview_df = scan_excel_table(
                file_path=file_path,
                sheet_name=sheet_name,
                header_row=header_row,
            )
        except Exception as exc:
            return pd.DataFrame(), header_row, [f"表格扫描失败：{exc}"]

        return preview_df, header_row, errors

    st.markdown("##### 选择特定表格区域")

    if n_rows < 2:
        return pd.DataFrame(), 0, ["当前工作表行数不足，无法选择包含表头和数据的区域。"]

    c1, c2 = st.columns(2)

    with c1:
        start_row_excel = int(
            st.number_input(
                "起始行号 / 表头行号（包含，从1开始）",
                min_value=1,
                max_value=max(n_rows - 1, 1),
                value=1,
                step=1,
                key=f"{key_prefix}_region_start_row",
            )
        )

        end_row_excel = int(
            st.number_input(
                "结束行号（包含，从1开始）",
                min_value=min(start_row_excel + 1, n_rows),
                max_value=n_rows,
                value=n_rows,
                step=1,
                key=f"{key_prefix}_region_end_row",
            )
        )

    with c2:
        start_col_excel = int(
            st.number_input(
                "起始列号（包含，从1开始，1=A列）",
                min_value=1,
                max_value=max(n_cols, 1),
                value=1,
                step=1,
                key=f"{key_prefix}_region_start_col",
            )
        )

        end_col_excel = int(
            st.number_input(
                "结束列号（包含，从1开始）",
                min_value=start_col_excel,
                max_value=n_cols,
                value=n_cols,
                step=1,
                key=f"{key_prefix}_region_end_col",
            )
        )

    # 页面上使用 Excel 习惯的 1-based 行列号；
    # 内部 scan_excel_table_region 使用 Python 0-based 且 end_row/end_col 不包含。
    start_row = start_row_excel - 1
    end_row = end_row_excel
    start_col = start_col_excel - 1
    end_col = end_col_excel
    header_row = start_row

    try:
        preview_df = scan_excel_table_region(
            file_path=file_path,
            sheet_name=sheet_name,
            start_row=start_row,
            end_row=end_row,
            start_col=start_col,
            end_col=end_col,
            header_row=header_row,
        )
    except Exception as exc:
        return pd.DataFrame(), header_row, [f"指定区域扫描失败：{exc}"]

    return preview_df, header_row, errors


def render_mapping_panel(
    file_path: str,
    sheet_name: str,
    detected_info: dict | None = None,
    default_header_row: int = 0,
    key_prefix: str = "mapping_panel",
) -> MappingPanelResult:
    st.subheader("② 手动确认字段映射")
    st.info("请在上方完成字段映射，并点击“确认字段映射”。")

    detected = detected_info or {}

    region_mode = st.radio(
        "数据区域",
        ["使用整张工作表的有效数据", "选择特定表格区域"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_region_mode",
    )

    preview_df, header_row, scan_errors = _scan_selected_region(
        file_path=file_path,
        sheet_name=sheet_name,
        region_mode=region_mode,
        default_header_row=default_header_row,
        key_prefix=key_prefix,
    )

    if scan_errors:
        for err in scan_errors:
            st.error(err)

        return MappingPanelResult(
            mapping=FieldMapping(),
            data_format="wide_table",
            selected_measure="Paid",
            header_row=header_row,
            preview_df=preview_df,
            confirmed=False,
            errors=scan_errors,
            warnings=[],
        )

    if preview_df.empty:
        st.warning("当前选择区域没有可预览的数据，请检查表头行或区域范围。")

        return MappingPanelResult(
            mapping=FieldMapping(),
            data_format="wide_table",
            selected_measure="Paid",
            header_row=header_row,
            preview_df=preview_df,
            confirmed=False,
            errors=["当前选择区域没有可预览的数据。"],
            warnings=[],
        )

    with st.expander("当前选择区域预览", expanded=False):
        st.dataframe(preview_df.head(20), use_container_width=True)

    visible_columns = _visible_columns(preview_df)
    col_options = [""] + visible_columns

    format_options = ["wide_table", "long_table"]
    format_labels = {
        "wide_table": "宽表 / 三角形",
        "long_table": "长表",
    }

    default_format = _get_value(detected, "data_format", default="wide_table")
    if default_format not in format_options:
        default_format = "wide_table"

    data_format = st.radio(
        "数据格式",
        format_options,
        format_func=lambda x: format_labels.get(x, x),
        index=format_options.index(default_format),
        horizontal=True,
        key=f"{key_prefix}_data_format",
    )

    ay_col = st.selectbox(
        "事故年列（必填）",
        col_options,
        index=_safe_index(col_options, _get_value(detected, "accident_year_col", default="")),
        key=f"{key_prefix}_ay_col",
    )

    dev_col = st.selectbox(
        "发展期列（长表必填，宽表可留空）",
        col_options,
        index=_safe_index(col_options, _get_value(detected, "development_col", default="")),
        key=f"{key_prefix}_dev_col",
    )

    type_col_name = st.selectbox(
        "数据口径列 / Type列（可选）",
        col_options,
        index=_safe_index(col_options, _get_value(detected, "type_col_name", default="")),
        key=f"{key_prefix}_type_col",
    )

    measure_col_value = str(_get_value(detected, "measure_col_value", default="Paid"))

    if type_col_name and type_col_name in preview_df.columns:
        unique_values = [
            str(x)
            for x in preview_df[type_col_name].dropna().unique()
            if str(x).strip()
        ]

        if unique_values:
            measure_col_value = st.selectbox(
                "目标口径",
                unique_values,
                index=_safe_index(unique_values, measure_col_value),
                key=f"{key_prefix}_measure_value_select",
            )
        else:
            st.warning("所选口径列没有有效取值。")
            measure_col_value = ""
    else:
        measure_col_value = st.text_input(
            "目标口径（没有口径列时可忽略）",
            value=measure_col_value,
            key=f"{key_prefix}_measure_value_text",
        )

    default_amount_cols = [
        c
        for c in _normalise_amount_cols(_get_value(detected, "amount_cols", default=[]))
        if c in visible_columns
    ]

    if data_format == "long_table":
        default_amount = default_amount_cols[0] if default_amount_cols else ""
        amount_col = st.selectbox(
            "金额列（长表必填，只能选择一个）",
            col_options,
            index=_safe_index(col_options, default_amount),
            key=f"{key_prefix}_amount_col_single",
        )
        amount_cols = [amount_col] if amount_col else []
    else:
        amount_cols = st.multiselect(
            "金额列（必填，可多选）",
            visible_columns,
            default=default_amount_cols,
            key=f"{key_prefix}_amount_cols_multi",
        )

    is_cumulative = st.checkbox(
        "是否为累计数据",
        value=bool(_get_value(detected, "is_cumulative", default=True)),
        key=f"{key_prefix}_is_cumulative",
    )

    mapping = FieldMapping(
        accident_year_col=ay_col,
        development_col=dev_col,
        amount_cols=amount_cols,
        type_col_name=type_col_name,
        measure_col_value=measure_col_value,
        is_cumulative=is_cumulative,
    )

    errors, warnings = _validate_mapping(preview_df, mapping, data_format)

    for err in errors:
        st.error(err)

    for warning in warnings:
        st.warning(warning)

    if not errors:
        st.success("字段映射完整，可以点击确认重新生成标准赔付三角形。")

    confirmed = st.button(
        "确认字段映射",
        type="primary",
        disabled=bool(errors),
        key=f"{key_prefix}_confirm",
    )

    return MappingPanelResult(
        mapping=mapping,
        data_format=data_format,
        selected_measure=measure_col_value,
        header_row=header_row,
        preview_df=preview_df,
        confirmed=confirmed,
        errors=errors,
        warnings=warnings,
    )