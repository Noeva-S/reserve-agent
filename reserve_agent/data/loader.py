from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from reserve_agent.data.detector import (
    RESERVING_FORMATS,
    ExcelFormat,
    detect_excel_format,
    normalise_label,
)
from reserve_agent.data.table_scanner import (
    TableRegion,
    find_candidate_table_regions,
    slice_region_with_header,
)


DEFAULT_WORKBOOK = "Chapter 08 - Data sets - Examples.xlsx"


@dataclass
class DataQualityReport:
    row_count: int
    claim_count: int
    accident_years: list[int]
    valuation_years: list[int]
    missing_values: int
    negative_amount_cells: int
    zero_claim_rows: int
    notes: list[str]


class UnsupportedExcelFormatError(ValueError):
    """Raised when no candidate table can be converted into a triangle."""

    detected_format: ExcelFormat = "unknown"


@dataclass
class DetectedTable:
    region: TableRegion
    table: pd.DataFrame
    format_name: ExcelFormat
    recognition_source: str = "rules"
    recognition_reason: str = ""


@dataclass
class ExcelLoadResult:
    source_table: pd.DataFrame
    triangle: pd.DataFrame
    format_name: ExcelFormat
    region: TableRegion
    candidates: list[DetectedTable]
    quality: DataQualityReport
    requested_sheet_name: str = ""
    source_sheet_name: str = ""
    recognition_source: str = "rules"
    warnings: tuple[str, ...] = ()


def find_default_workbook(base_dir: str | Path = ".") -> Path:
    base = Path(base_dir)
    candidate = base / DEFAULT_WORKBOOK
    if candidate.exists():
        return candidate
    workbooks = list(base.glob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError("未找到 Excel 数据文件。")
    return workbooks[0]


def list_excel_sheets(file_path: str | Path) -> list[str]:
    with pd.ExcelFile(
        file_path,
        engine="openpyxl",
        engine_kwargs={"keep_links": False},
    ) as workbook:
        return list(workbook.sheet_names)


def _base_sheet_name(name: str) -> str:
    lowered = str(name).strip().lower()
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)\s_-]*", "", lowered).strip()


def choose_default_sheet(sheet_names: list[str]) -> int:
    """Prefer a Claims data sheet even when it has a numeric prefix."""

    if not sheet_names:
        return 0
    for index, name in enumerate(sheet_names):
        if _base_sheet_name(name) == "claims data":
            return index
    for index, name in enumerate(sheet_names):
        if "claims data" in _base_sheet_name(name):
            return index
    return 0


def load_claims_snapshot(file_path: str | Path, sheet_name: str = "Claims data") -> pd.DataFrame:
    """Load the original teaching workbook's snapshot-style claims table."""

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    raw = raw.dropna(axis=1, how="all").dropna(how="all")

    rename_map = {}
    for col in raw.columns:
        if isinstance(col, str):
            clean = col.strip()
            rename_map[col] = "" if clean.startswith("Unnamed") else clean
    raw = raw.rename(columns=rename_map)
    if "" in raw.columns:
        raw = raw.drop(columns=[""])

    required = {"Claim ID", "Loss Year", "Type"}
    missing = required.difference(set(raw.columns))
    if missing:
        raise ValueError(f"工作表缺少必要字段：{', '.join(sorted(missing))}")

    raw = raw[raw["Claim ID"].notna()].copy()
    raw["Loss Year"] = pd.to_numeric(raw["Loss Year"], errors="coerce").astype("Int64")
    raw["Type"] = raw["Type"].astype(str).str.strip()
    return raw


def valuation_year_columns(df: pd.DataFrame) -> list[int]:
    years: list[int] = []
    for col in df.columns:
        if isinstance(col, (int, np.integer)):
            years.append(int(col))
        elif isinstance(col, float) and col.is_integer():
            years.append(int(col))
        elif isinstance(col, str) and col.strip().isdigit():
            years.append(int(col.strip()))
    return sorted(set(years))


def build_cumulative_triangle(
    claims_df: pd.DataFrame,
    measure: str = "Paid",
    accident_year_col: str = "Loss Year",
) -> pd.DataFrame:
    """Build an accident-year by development-age cumulative triangle."""

    years = valuation_year_columns(claims_df)
    if not years:
        raise ValueError("未识别到评估年份列。")

    measure_df = claims_df[claims_df["Type"].astype(str).str.lower() == measure.lower()].copy()
    if measure_df.empty:
        available = ", ".join(sorted(claims_df["Type"].dropna().astype(str).unique()))
        raise ValueError(f"未找到 Type={measure} 的记录。可用 Type：{available}")

    long_df = measure_df.melt(
        id_vars=["Claim ID", accident_year_col],
        value_vars=years,
        var_name="valuation_year",
        value_name="amount",
    )
    long_df["valuation_year"] = pd.to_numeric(long_df["valuation_year"], errors="coerce")
    long_df["amount"] = pd.to_numeric(long_df["amount"], errors="coerce")
    long_df[accident_year_col] = pd.to_numeric(long_df[accident_year_col], errors="coerce")
    long_df = long_df.dropna(subset=[accident_year_col, "valuation_year"])
    long_df["development"] = (long_df["valuation_year"] - long_df[accident_year_col]).astype(int)
    long_df = long_df[long_df["development"] >= 0]

    latest_valuation_year = int(max(years))
    long_df = long_df[long_df[accident_year_col] + long_df["development"] <= latest_valuation_year]

    grouped = (
        long_df.groupby([accident_year_col, "development"], as_index=False)["amount"]
        .sum(min_count=1)
        .sort_values([accident_year_col, "development"])
    )

    triangle = grouped.pivot(index=accident_year_col, columns="development", values="amount")
    triangle = triangle.sort_index().sort_index(axis=1)
    triangle.index = triangle.index.astype(int)
    triangle.columns = triangle.columns.astype(int)
    return triangle.astype(float)


def latest_diagonal(cumulative_triangle: pd.DataFrame) -> pd.Series:
    latest = {}
    for ay, row in cumulative_triangle.iterrows():
        observed = row.dropna()
        latest[ay] = np.nan if observed.empty else observed.iloc[-1]
    return pd.Series(latest, name="latest_cumulative")


def load_exposure_data(file_path: str | Path, sheet_name: str = "Exposure data (PL)") -> pd.DataFrame:
    """Load a simple policy-year exposure table when available."""

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    except Exception:
        return pd.DataFrame(columns=["Policy year", "Exposure"])

    candidates = []
    for year_col in ["Policy year.1", "Policy year"]:
        for exposure_col in [
            "Turnover (拢m) - revalued @ 4% p.a.",
            "Turnover (x 1€m) - revalued @ 4% p.a.",
            "Turnover (x 1鈧琺) - revalued @ 4% p.a.",
            "Turnover (x 1閳х惡) - revalued @ 4% p.a.",
            "Employee Numbers",
            "Employee Numbers.1",
        ]:
            if year_col in df.columns and exposure_col in df.columns:
                temp = df[[year_col, exposure_col]].copy()
                temp.columns = ["Policy year", "Exposure"]
                temp["Policy year"] = pd.to_numeric(temp["Policy year"], errors="coerce")
                temp["Exposure"] = pd.to_numeric(temp["Exposure"], errors="coerce")
                temp = temp.dropna()
                if len(temp) > 0:
                    candidates.append(temp)
    if not candidates:
        return pd.DataFrame(columns=["Policy year", "Exposure"])

    result = candidates[0].drop_duplicates("Policy year").sort_values("Policy year")
    result["Policy year"] = result["Policy year"].astype(int)
    return result


def quality_report(claims_df: pd.DataFrame, triangle: pd.DataFrame) -> DataQualityReport:
    years = valuation_year_columns(claims_df)
    amount_cells = claims_df[years].apply(pd.to_numeric, errors="coerce") if years else pd.DataFrame()
    negative_amount_cells = int((amount_cells < 0).sum().sum()) if not amount_cells.empty else 0
    zero_claim_rows = int((amount_cells.fillna(0).sum(axis=1) == 0).sum()) if not amount_cells.empty else 0
    missing_values = int(claims_df.isna().sum().sum())
    notes = []

    if negative_amount_cells:
        notes.append("存在负赔款单元格，可能来自追偿、冲回或数据修正，建模前应单独复核。")
    if zero_claim_rows:
        notes.append("存在全发展期金额为 0 的赔案记录，系统已保留但在解释中提示关注。")
    if triangle.shape[0] < 5:
        notes.append("事故年数量偏少，发展因子稳定性有限。")
    if triangle.shape[1] and triangle.iloc[:, -1].notna().sum() < 2:
        notes.append("最末发展期观察不足，尾部因子需要谨慎判断。")
    if not notes:
        notes.append("未发现会阻断建模的严重数据质量问题。")

    return DataQualityReport(
        row_count=int(len(claims_df)),
        claim_count=int(claims_df["Claim ID"].nunique()) if "Claim ID" in claims_df.columns else int(len(triangle.index)),
        accident_years=[int(x) for x in triangle.index.tolist()],
        valuation_years=[int(x) for x in years],
        missing_values=missing_values,
        negative_amount_cells=negative_amount_cells,
        zero_claim_rows=zero_claim_rows,
        notes=notes,
    )


def _generic_quality_report(
    source_table: pd.DataFrame,
    triangle: pd.DataFrame,
    format_name: ExcelFormat,
) -> DataQualityReport:
    numeric_triangle = triangle.apply(pd.to_numeric, errors="coerce")
    negative_cells = int((numeric_triangle < 0).sum().sum())
    zero_rows = int((numeric_triangle.fillna(0).sum(axis=1) == 0).sum())
    missing_values = int(numeric_triangle.isna().sum().sum())
    accident_years = [int(year) for year in triangle.index]
    valuation_years = sorted(
        {
            int(accident_year) + int(development)
            for accident_year, row in triangle.iterrows()
            for development, value in row.items()
            if pd.notna(value)
        }
    )
    if not valuation_years:
        valuation_years = accident_years.copy()

    format_labels = {
        "triangle": "事故年 x 发展期三角形",
        "long_table": "事故年/发展期/金额长表",
        "claims_snapshot": "赔案快照明细",
        "policy_data": "保单数据",
        "exposure_data": "暴露量数据",
        "unknown": "未知格式",
    }
    notes = [f"系统自动识别为{format_labels.get(format_name, format_name)}格式。"]
    if any("delayday" in normalise_label(column) for column in source_table.columns):
        notes.append("原表按天记录报告延迟，系统已按 365.25 天向下归入年度发展期。")
    if negative_cells:
        notes.append("三角形中存在负值，可能来自追偿、冲回或数据修正。")
    if zero_rows:
        notes.append("存在所有已提供发展期金额均为 0 的事故年。")
    if triangle.shape[0] < 5:
        notes.append("事故年数量偏少，发展因子稳定性有限。")
    if triangle.shape[1] and triangle.iloc[:, -1].notna().sum() < 2:
        notes.append("最末发展期观察不足，尾部因子需要谨慎判断。")

    return DataQualityReport(
        row_count=int(len(source_table)),
        claim_count=int(len(triangle.index)),
        accident_years=accident_years,
        valuation_years=valuation_years,
        missing_values=missing_values,
        negative_amount_cells=negative_cells,
        zero_claim_rows=zero_rows,
        notes=notes,
    )


def scan_excel_sheet(file_path: str | Path, sheet_name: str) -> tuple[pd.DataFrame, list[DetectedTable]]:
    """Read a raw worksheet, find candidate regions, and classify each one."""

    raw = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
        engine_kwargs={"keep_links": False},
    )
    regions = find_candidate_table_regions(raw)
    candidates = []
    for region in regions:
        table = slice_region_with_header(raw, region)
        candidates.append(DetectedTable(region=region, table=table, format_name=detect_excel_format(table)))
    return raw, candidates


def _adapt_first_candidate(
    candidates: list[DetectedTable],
    *,
    measure: str,
    is_cumulative: bool,
) -> tuple[DetectedTable, pd.DataFrame] | None:
    from reserve_agent.data.adapters import adapt_to_triangle

    for candidate in candidates:
        if candidate.format_name not in RESERVING_FORMATS:
            continue
        try:
            triangle = adapt_to_triangle(
                candidate.table,
                candidate.format_name,
                measure=measure,
                is_cumulative=is_cumulative,
            )
        except (TypeError, ValueError):
            continue
        if not triangle.empty:
            return candidate, triangle
    return None


def _api_assisted_candidate(
    candidates: list[DetectedTable],
    *,
    sheet_name: str,
    measure: str,
    is_cumulative: bool,
    api_key: str,
) -> tuple[DetectedTable, pd.DataFrame]:
    from reserve_agent.data.api_detector import apply_api_mapping, request_api_detection

    result = request_api_detection(
        [candidate.table for candidate in candidates],
        sheet_name=sheet_name,
        measure=measure,
        api_key=api_key,
    )
    original = candidates[result.candidate_index]
    mapped = DetectedTable(
        region=original.region,
        table=apply_api_mapping(original.table, result),
        format_name=result.format_name,
        recognition_source="api",
        recognition_reason=result.reason,
    )
    adapted = _adapt_first_candidate([mapped], measure=measure, is_cumulative=is_cumulative)
    if adapted is None:
        raise UnsupportedExcelFormatError("API 返回的字段映射无法转换为有效赔付三角。")
    return adapted


def _fallback_sheet_order(sheet_names: list[str], requested_sheet: str) -> list[str]:
    def rank(name: str) -> tuple[int, int]:
        base = _base_sheet_name(name)
        if base == "claims data":
            priority = 0
        elif "claims" in base or "claim" in base or "赔案" in base or "索赔" in base:
            priority = 1
        elif any(word in base for word in ("triangle", "projection", "reserve", "三角", "准备金")):
            priority = 2
        elif any(word in base for word in ("disclaimer", "说明", "exposure", "policy data")):
            priority = 4
        else:
            priority = 3
        return priority, sheet_names.index(name)

    return sorted((name for name in sheet_names if name != requested_sheet), key=rank)


def load_excel_to_triangle(
    file_path: str | Path,
    sheet_name: str,
    *,
    measure: str = "Paid",
    is_cumulative: bool = True,
    api_key: str | None = None,
    fallback_to_other_sheets: bool = True,
) -> ExcelLoadResult:
    """Detect a reserving table, using API/schema and workbook fallbacks when needed."""

    requested_sheet = sheet_name
    warnings: list[str] = []
    api_error: str | None = None
    _, requested_candidates = scan_excel_sheet(file_path, requested_sheet)
    adapted = _adapt_first_candidate(requested_candidates, measure=measure, is_cumulative=is_cumulative)

    locally_recognised = any(candidate.format_name != "unknown" for candidate in requested_candidates)
    if adapted is None and api_key and not locally_recognised:
        try:
            adapted = _api_assisted_candidate(
                requested_candidates,
                sheet_name=requested_sheet,
                measure=measure,
                is_cumulative=is_cumulative,
                api_key=api_key,
            )
            requested_candidates.append(adapted[0])
            warnings.append("本地规则未能完成字段映射，本次使用 DeepSeek API 辅助识别。")
        except Exception as exc:
            api_error = str(exc)

    source_sheet = requested_sheet
    candidates = requested_candidates
    if adapted is None and fallback_to_other_sheets:
        for candidate_sheet in _fallback_sheet_order(list_excel_sheets(file_path), requested_sheet):
            try:
                _, other_candidates = scan_excel_sheet(file_path, candidate_sheet)
            except Exception:
                continue
            other_adapted = _adapt_first_candidate(
                other_candidates,
                measure=measure,
                is_cumulative=is_cumulative,
            )
            if other_adapted is not None:
                adapted = other_adapted
                source_sheet = candidate_sheet
                candidates = other_candidates
                warnings.append(
                    f"所选工作表“{requested_sheet}”不能直接形成赔付发展三角，"
                    f"系统已自动改用“{candidate_sheet}”。"
                )
                break

    if adapted is None:
        detected = sorted(
            {
                candidate.format_name
                for candidate in requested_candidates
                if candidate.format_name != "unknown"
            }
        )
        details = (
            f"本地规则识别到：{', '.join(detected)}；但这些数据不能单独形成赔付发展三角。"
            if detected
            else "本地规则未找到可建模的数据表。"
        )
        if api_error:
            details += f" API 辅助识别也未成功：{api_error}"
        elif not api_key:
            details += " 如需启用 API 兜底，请打开 DeepSeek API 开关并配置 DEEPSEEK_API_KEY。"
        raise UnsupportedExcelFormatError(
            f"无法从工作表“{requested_sheet}”或同工作簿其他工作表生成赔付三角。{details}"
        )

    selected, triangle = adapted
    if selected.format_name == "claims_snapshot":
        try:
            quality = quality_report(selected.table, triangle)
        except Exception:
            quality = _generic_quality_report(selected.table, triangle, selected.format_name)
    else:
        quality = _generic_quality_report(selected.table, triangle, selected.format_name)

    return ExcelLoadResult(
        source_table=selected.table,
        triangle=triangle,
        format_name=selected.format_name,
        region=selected.region,
        candidates=candidates,
        quality=quality,
        requested_sheet_name=requested_sheet,
        source_sheet_name=source_sheet,
        recognition_source=selected.recognition_source,
        warnings=tuple(warnings),
    )


def triangle_to_display(triangle: pd.DataFrame) -> pd.DataFrame:
    display = triangle.copy()
    display.index.name = "Accident Year"
    display.columns = [f"Dev {int(c)}" for c in display.columns]
    return display


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def summarize_claims_by_year(claims_df: pd.DataFrame, measures: Iterable[str] = ("Paid", "Incurred")) -> pd.DataFrame:
    rows = []
    for measure in measures:
        tri = build_cumulative_triangle(claims_df, measure=measure)
        latest = latest_diagonal(tri)
        for ay, val in latest.items():
            rows.append({"Measure": measure, "Accident Year": ay, "Latest": val})
    return pd.DataFrame(rows)
