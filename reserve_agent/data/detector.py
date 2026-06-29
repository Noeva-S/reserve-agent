from __future__ import annotations

import math
import re
from typing import Any, Literal

import pandas as pd


ExcelFormat = Literal["claims_snapshot", "triangle", "long_table", "unknown"]


ROLE_ALIASES: dict[str, set[str]] = {
    "claim_id": {"claimid", "claimnumber", "claimno", "claim", "赔案号", "赔案编号", "索赔编号"},
    "accident_year": {
        "lossyear",
        "accidentyear",
        "originyear",
        "policyyear",
        "policyyearshifted",
        "ay",
        "事故年",
        "出险年",
        "保单年",
    },
    "measure": {"type", "measure", "datatype", "口径", "类型"},
    "development": {
        "development",
        "developmentage",
        "developmentperiod",
        "dev",
        "devage",
        "dy",
        "delayyears",
        "delayyear",
        "delaythirds",
        "delayshifted",
        "发展期",
        "发展年",
        "进展期",
        "延迟期",
    },
    "amount": {
        "amount",
        "value",
        "loss",
        "paid",
        "paidamount",
        "incurred",
        "totalincurred",
        "overallamount",
        "overallamountdestinationcurrency",
        "overallamountrevalued",
        "cumulativeamount",
        "cumulativeloss",
        "金额",
        "赔款",
        "赔款金额",
        "累计赔款",
        "累计赔款金额",
    },
}


def normalise_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    return re.sub(r"[\s_\-./\\()（）:：#]+", "", text)


def find_role_column(df: pd.DataFrame, role: str) -> Any | None:
    aliases = ROLE_ALIASES.get(role, set())
    for column in df.columns:
        if normalise_label(column) in aliases:
            return column
    return None


def parse_development_label(value: Any) -> int | None:
    """Parse explicit development labels such as 0, Dev 1, 12 months.

    The function intentionally avoids loose rules such as "any number plus the
    word year", because exposure columns like "Turnover (x 1m) - policy year"
    would otherwise be misclassified as development columns.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer() and numeric >= 0:
            return int(numeric)
        return None

    text = str(value).strip().lower()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))

    explicit_prefix = re.fullmatch(
        r"(?:dev(?:elopment)?(?:\s*(?:age|period))?|age|delay|year|month|发展(?:期|年)?|进展期|延迟期)"
        r"\s*[:#._\-/]*\s*(\d+)(?:\.0+)?\s*",
        text,
    )
    if explicit_prefix:
        return int(explicit_prefix.group(1))

    explicit_suffix = re.fullmatch(r"(\d+)(?:\.0+)?\s*(?:months?|years?|quarters?|月|年|期)", text)
    if explicit_suffix:
        return int(explicit_suffix.group(1))

    return None


def detect_excel_format(df: pd.DataFrame) -> ExcelFormat:
    """Classify a candidate table after table-region scanning."""

    if df is None or df.empty:
        return "unknown"

    claim_col = find_role_column(df, "claim_id")
    accident_col = find_role_column(df, "accident_year")
    measure_col = find_role_column(df, "measure")
    development_col = find_role_column(df, "development")
    amount_col = find_role_column(df, "amount")

    if claim_col is not None and accident_col is not None and measure_col is not None:
        return "claims_snapshot"

    if accident_col is not None and development_col is not None and amount_col is not None:
        return "long_table"

    # Claim-level long tables often contain Claim ID, Policy Year, Delay, Paid
    # and Total incurred, without a generic "Amount" column.
    if claim_col is not None and accident_col is not None and development_col is not None:
        amount_like = [column for column in df.columns if normalise_label(column) in ROLE_ALIASES["amount"]]
        if amount_like:
            return "long_table"

    index_is_accident_year = normalise_label(df.index.name) in ROLE_ALIASES["accident_year"]
    has_accident_axis = accident_col is not None or index_is_accident_year
    development_columns = [
        column
        for column in df.columns
        if column != accident_col and parse_development_label(column) is not None
    ]
    if has_accident_axis and len(development_columns) >= 2:
        return "triangle"

    return "unknown"
