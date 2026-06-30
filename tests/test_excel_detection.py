from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from reserve_agent.data.detector import detect_excel_format, parse_development_label
from reserve_agent.data.loader import (
    UnsupportedExcelFormatError,
    choose_default_sheet,
    load_excel_to_triangle,
)
from reserve_agent.data.table_scanner import find_candidate_table_regions, slice_region_with_header


def _write_raw_workbook(path: Path, rows: list[list[object]], sheet_name: str = "Data") -> None:
    width = max(len(row) for row in rows)
    padded = [row + [None] * (width - len(row)) for row in rows]
    pd.DataFrame(padded).to_excel(path, sheet_name=sheet_name, header=False, index=False)


class ExcelDetectionTests(unittest.TestCase):
    def test_claims_header_after_three_description_rows(self) -> None:
        rows = [
            ["赔案数据测试文件"],
            ["说明：前三行不是数据表"],
            ["单位：元"],
            ["Claim ID", "Loss Year", "Type", 2007, 2008, 2009],
            [1, 2007, "Paid", 100, 150, 180],
            [2, 2008, "Paid", None, 80, 120],
            [1, 2007, "Incurred", 130, 170, 190],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claims_with_notes.xlsx"
            _write_raw_workbook(path, rows)
            result = load_excel_to_triangle(path, "Data", measure="Paid")

        self.assertEqual(result.format_name, "claims_snapshot")
        self.assertEqual(result.region.header_row, 3)
        self.assertEqual(result.triangle.loc[2007, 0], 100)
        self.assertEqual(result.triangle.loc[2008, 1], 120)

    def test_triangle_with_title_and_blank_row(self) -> None:
        rows = [
            ["累计赔款三角"],
            [None],
            ["事故年", 0, 1, 2],
            [2021, 100, 150, 180],
            [2022, 120, 170, None],
            [2023, 130, None, None],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triangle.xlsx"
            _write_raw_workbook(path, rows)
            result = load_excel_to_triangle(path, "Data")

        self.assertEqual(result.format_name, "triangle")
        self.assertEqual(list(result.triangle.columns), [0, 1, 2])
        self.assertEqual(result.triangle.loc[2022, 1], 170)

    def test_long_table_below_an_unrelated_table(self) -> None:
        rows = [
            ["参数", "取值"],
            ["币种", "CNY"],
            [None, None],
            ["Accident Year", "Development", "Amount"],
            [2021, 0, 100],
            [2021, 1, 150],
            [2022, 0, 120],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_table.xlsx"
            _write_raw_workbook(path, rows)
            result = load_excel_to_triangle(path, "Data")

        self.assertEqual(result.format_name, "long_table")
        self.assertEqual(result.region.header_row, 3)
        self.assertEqual(result.triangle.loc[2021, 1], 150)

    def test_detector_returns_unknown(self) -> None:
        unknown = pd.DataFrame({"名称": ["测试"], "备注": ["无法建模"]})
        self.assertEqual(detect_excel_format(unknown), "unknown")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.xlsx"
            unknown.to_excel(path, index=False)
            with self.assertRaises(UnsupportedExcelFormatError):
                load_excel_to_triangle(path, "Sheet1")

    def test_scanner_and_slicer_promote_real_header(self) -> None:
        raw = pd.DataFrame(
            [
                ["标题", None, None],
                [None, None, None],
                ["事故年", "发展期", "金额"],
                [2022, 0, 100],
            ]
        )
        regions = find_candidate_table_regions(raw)
        table = slice_region_with_header(raw, regions[0])
        self.assertEqual(regions[0].header_row, 2)
        self.assertEqual(list(table.columns), ["事故年", "发展期", "金额"])

    def test_numbered_claims_sheet_is_selected_by_default(self) -> None:
        sheets = ["00. DISCLAIMER", "1. Exposure data", "2. Claims data"]
        self.assertEqual(choose_default_sheet(sheets), 2)

    def test_claim_level_long_table_uses_shifted_axes_and_cumulative_amounts(self) -> None:
        rows = [
            ["课程工作簿赔案明细"],
            [None],
            [
                "Claim ID",
                "Policy Year",
                "Paid",
                "Total incurred",
                "Policy Year (shifted)",
                "Delay (years)",
                "Delay (shifted)",
            ],
            ["C1", 2020, 100, 150, 2019, 0, 0],
            ["C2", 2020, 50, 80, 2019, 1, 1],
            ["C3", 2021, 70, 90, 2020, 0, 0],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim_level.xlsx"
            _write_raw_workbook(path, rows)
            paid = load_excel_to_triangle(path, "Data", measure="Paid")
            incurred = load_excel_to_triangle(path, "Data", measure="Incurred")

        self.assertEqual(paid.format_name, "long_table")
        self.assertEqual(paid.region.header_row, 2)
        self.assertEqual(paid.triangle.loc[2019, 0], 100)
        self.assertEqual(paid.triangle.loc[2019, 1], 150)
        self.assertEqual(incurred.triangle.loc[2019, 0], 150)
        self.assertEqual(incurred.triangle.loc[2019, 1], 230)

    def test_claim_detail_with_delay_days_is_bucketed_to_development_years(self) -> None:
        rows = [
            ["Claim ID", "Policy Year", "Paid", "Total incurred", "Delay (days)"],
            ["C1", 2020, 100, 150, 0],
            ["C2", 2020, 50, 80, 400],
            ["C3", 2021, 70, 90, 20],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claims_delay_days.xlsx"
            _write_raw_workbook(path, rows)
            result = load_excel_to_triangle(path, "Data", measure="Paid")

        self.assertEqual(result.format_name, "long_table")
        self.assertEqual(list(result.triangle.columns), [0, 1])
        self.assertEqual(result.triangle.loc[2020, 0], 100)
        self.assertEqual(result.triangle.loc[2020, 1], 150)
        self.assertTrue(any("365.25" in note for note in result.quality.notes))

    def test_policy_data_is_recognised_but_claims_sheet_is_used_for_model(self) -> None:
        policy = pd.DataFrame(
            {
                "Policy ID": ["P1", "P2"],
                "Inception Date": ["2020-01-01", "2021-01-01"],
                "Net Premium": [1000, 1200],
            }
        )
        self.assertEqual(detect_excel_format(policy), "policy_data")

        claims = pd.DataFrame(
            {
                "Claim ID": ["C1", "C2"],
                "Loss Year": [2020, 2021],
                "Type": ["Paid", "Paid"],
                2020: [100, None],
                2021: [150, 80],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                policy.to_excel(writer, sheet_name="Policy data", index=False)
                claims.to_excel(writer, sheet_name="Claims data", index=False)
            with patch("reserve_agent.agent.llm_client.call_deepseek") as mocked_api:
                result = load_excel_to_triangle(
                    path,
                    "Policy data",
                    measure="Paid",
                    api_key="test-key",
                )

        self.assertEqual(result.requested_sheet_name, "Policy data")
        self.assertEqual(result.source_sheet_name, "Claims data")
        self.assertEqual(result.format_name, "claims_snapshot")
        self.assertTrue(result.warnings)
        mocked_api.assert_not_called()

    def test_api_fallback_maps_unknown_columns_without_sending_cell_values(self) -> None:
        rows = [
            ["Origin Period", "Lag Bucket", "Paid Cash", "Claimant Name"],
            [2020, 0, 100, "Alice Sensitive"],
            [2020, 1, 150, "Bob Sensitive"],
            [2021, 0, 80, "Carol Sensitive"],
        ]
        response = {
            "candidate_index": 0,
            "format_name": "long_table",
            "column_mapping": {
                "accident_year": "Origin Period",
                "development": "Lag Bucket",
                "amount": "Paid Cash",
            },
            "development_unit": "years",
            "confidence": 0.96,
            "reason": "列结构符合事故期、发展期、金额长表。",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_fallback.xlsx"
            _write_raw_workbook(path, rows)
            with patch("reserve_agent.agent.llm_client.call_deepseek", return_value=json.dumps(response)) as mocked:
                result = load_excel_to_triangle(
                    path,
                    "Data",
                    measure="Paid",
                    api_key="test-key",
                    fallback_to_other_sheets=False,
                )

        messages = mocked.call_args.args[0]
        prompt_text = "\n".join(message["content"] for message in messages)
        self.assertNotIn("Alice Sensitive", prompt_text)
        self.assertEqual(result.recognition_source, "api")
        self.assertEqual(result.triangle.loc[2020, 1], 150)

    def test_exposure_column_with_currency_unit_is_not_development(self) -> None:
        exposure = pd.DataFrame(
            {
                "Policy year": [2019, 2020],
                "Turnover (x 1€m) - aligned to policy year": [10, 12],
                "Turnover (x 1€m) - revalued @ 4% p.a.": [11, 13],
            }
        )
        self.assertIsNone(parse_development_label("Turnover (x 1€m) - aligned to policy year"))
        self.assertEqual(detect_excel_format(exposure), "exposure_data")

    def test_chapter_08_workbook_claims_pl_sheet(self) -> None:
        workbook = Path(__file__).resolve().parents[1] / "Chapter 08 - Data sets - Examples.xlsx"
        if not workbook.exists():
            self.skipTest("Chapter 08 workbook is not available")

        result = load_excel_to_triangle(workbook, "Claims data (PL)", measure="Paid")
        self.assertEqual(result.format_name, "long_table")
        self.assertEqual(result.region.header_row, 11)
        self.assertGreaterEqual(result.triangle.shape[0], 8)
        self.assertGreaterEqual(result.triangle.shape[1], 5)

    def test_chapter_13_workbook_claims_sheet(self) -> None:
        workbook = Path(__file__).resolve().parents[1] / "Chapter 13a - IBNR (triangle-based) v4.xlsx"
        if not workbook.exists():
            self.skipTest("Chapter 13 workbook is not available")

        result = load_excel_to_triangle(workbook, "2. Claims data", measure="Paid")
        self.assertEqual(result.format_name, "long_table")
        self.assertEqual(result.region.header_row, 11)
        self.assertGreaterEqual(result.triangle.shape[0], 8)
        self.assertGreaterEqual(result.triangle.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
