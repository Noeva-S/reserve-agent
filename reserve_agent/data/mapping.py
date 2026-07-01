from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldMapping:
    """用户确认后的字段映射。

    兼容两条链路：
    1. 旧版自动识别链路使用 amount_col / measure_col；
    2. 队友第 3 部分手动映射链路使用 amount_cols / type_col_name / measure_col_value。
    """

    accident_year_col: str | None = ""
    development_col: str | None = ""
    amount_cols: list[str] = field(default_factory=list)
    type_col_name: str | None = ""
    measure_col_value: str | None = "Paid"
    is_cumulative: bool = True

    # 旧字段名兼容：允许 FieldMapping(amount_col="Paid") 这样的历史调用继续工作。
    amount_col: str | None = None
    measure_col: str | None = None

    def __post_init__(self) -> None:
        if self.amount_col and not self.amount_cols:
            self.amount_cols = [str(self.amount_col)]
        elif self.amount_cols and not self.amount_col:
            self.amount_col = str(self.amount_cols[0])

        if self.measure_col and not self.type_col_name:
            self.type_col_name = self.measure_col
        elif self.type_col_name and not self.measure_col:
            self.measure_col = self.type_col_name

        self.amount_cols = [str(col) for col in (self.amount_cols or []) if str(col).strip()]
        self.accident_year_col = "" if self.accident_year_col is None else str(self.accident_year_col)
        self.development_col = "" if self.development_col is None else str(self.development_col)
        self.type_col_name = "" if self.type_col_name is None else str(self.type_col_name)
        self.measure_col = "" if self.measure_col is None else str(self.measure_col)
        self.measure_col_value = "" if self.measure_col_value is None else str(self.measure_col_value)

    def is_valid(self) -> bool:
        """手动映射至少要有事故年列和一个金额列。长表的发展期由 UI 额外校验。"""
        return bool(self.accident_year_col and self.amount_cols)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accident_year_col": self.accident_year_col,
            "development_col": self.development_col,
            "amount_cols": list(self.amount_cols),
            "amount_col": self.amount_col or (self.amount_cols[0] if self.amount_cols else ""),
            "type_col_name": self.type_col_name,
            "measure_col": self.measure_col,
            "measure_col_value": self.measure_col_value,
            "is_cumulative": self.is_cumulative,
        }


def mapping_is_complete(mapping: FieldMapping) -> bool:
    """兼容旧版校验函数。"""
    return mapping.is_valid() and bool(mapping.development_col or len(mapping.amount_cols) >= 1)
