#!/usr/bin/env python3
"""
CSV MCP Server - Simple utilities for understanding and operating on CSV files.

This server provides tools to inspect CSV file structure, analyze column data,
get unique values, compute statistics, and create filtered/transformed CSVs for
downstream processing (e.g., visualization with ggplot via R MCP).
"""

import os
import csv
import subprocess
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("csv_mcp")

# Constants
CHARACTER_LIMIT = 25000  # Maximum response size in characters
MAX_SAMPLE_ROWS = 100    # Maximum rows for sampling


# Pydantic Models for Input Validation

class InspectStructureInput(BaseModel):
    """Input model for inspecting CSV file structure."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to CSV file (relative to current directory or absolute)",
        min_length=1
    )
    sample_rows: int = Field(
        default=5,
        description="Number of sample rows to display from beginning of file",
        ge=1,
        le=MAX_SAMPLE_ROWS
    )


class ColumnInfoInput(BaseModel):
    """Input model for getting detailed column information."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to CSV file (relative to current directory or absolute)",
        min_length=1
    )
    column_name: str = Field(
        ...,
        description="Name of the column to analyze (exact match, case-sensitive)",
        min_length=1
    )
    max_unique: int = Field(
        default=100,
        description="Maximum number of unique values to display",
        ge=1,
        le=1000
    )


class UniqueValuesInput(BaseModel):
    """Input model for getting unique values in a column."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to CSV file (relative to current directory or absolute)",
        min_length=1
    )
    column_name: str = Field(
        ...,
        description="Name of the column to get unique values from (exact match, case-sensitive)",
        min_length=1
    )
    limit: Optional[int] = Field(
        default=None,
        description="Optional limit on number of unique values to return",
        ge=1,
        le=10000
    )


class SummaryStatisticsInput(BaseModel):
    """Input model for getting summary statistics."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to CSV file (relative to current directory or absolute)",
        min_length=1
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific column names to analyze. If not provided, analyzes all numeric columns",
        max_items=100
    )


class FilterTransformInput(BaseModel):
    """Input model for filtering and transforming CSV files."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to input CSV file (relative to current directory or absolute)",
        min_length=1
    )
    output_path: str = Field(
        ...,
        description="Path for output CSV file (relative to current directory or absolute)",
        min_length=1
    )
    select_columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of column names to include in output. If not provided, includes all columns",
        max_items=100
    )
    filter_conditions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of simple filter conditions in format 'column_name=value' or 'column_name!=value'",
        max_items=50
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Optional column name to sort by"
    )


class PreviewRowsInput(BaseModel):
    """Input model for previewing specific rows."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_path: str = Field(
        ...,
        description="Path to CSV file (relative to current directory or absolute)",
        min_length=1
    )
    start_row: int = Field(
        default=1,
        description="Starting row number (1-indexed, excluding header)",
        ge=1
    )
    num_rows: int = Field(
        default=10,
        description="Number of rows to display",
        ge=1,
        le=100
    )


# Shared Utility Functions

def _resolve_path(file_path: str) -> Path:
    """Resolve and validate file path."""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _validate_csv_file(file_path: str) -> tuple[bool, str, Path]:
    """
    Validate that file exists and is readable.

    Returns:
        tuple of (success: bool, error_message: str, resolved_path: Path)
    """
    try:
        path = _resolve_path(file_path)

        if not path.exists():
            return False, f"Error: File not found at '{path}'. Please check the path is correct.", path

        if not path.is_file():
            return False, f"Error: '{path}' is not a file.", path

        if not os.access(path, os.R_OK):
            return False, f"Error: Cannot read file '{path}'. Permission denied.", path

        return True, "", path

    except Exception as e:
        return False, f"Error: Invalid path '{file_path}': {str(e)}", Path(file_path)


def _read_csv_headers(file_path: Path) -> tuple[bool, str, List[str]]:
    """
    Read CSV headers.

    Returns:
        tuple of (success: bool, error_message: str, headers: List[str])
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return True, "", headers
    except StopIteration:
        return False, "Error: CSV file is empty.", []
    except Exception as e:
        return False, f"Error: Failed to read CSV headers: {str(e)}", []


def _get_column_index(headers: List[str], column_name: str) -> tuple[bool, str, int]:
    """
    Get column index by name.

    Returns:
        tuple of (success: bool, error_message: str, index: int)
    """
    try:
        index = headers.index(column_name)
        return True, "", index
    except ValueError:
        available = ", ".join([f"'{h}'" for h in headers[:10]])
        if len(headers) > 10:
            available += f", ... ({len(headers) - 10} more)"
        return False, f"Error: Column '{column_name}' not found. Available columns: {available}", -1


def _infer_data_type(values: List[str]) -> str:
    """
    Infer data type from a sample of values.

    Returns: 'integer', 'float', 'boolean', or 'string'
    """
    if not values:
        return 'string'

    # Try integer
    int_count = 0
    float_count = 0
    bool_count = 0

    for val in values:
        val = val.strip()
        if not val:
            continue

        # Check boolean
        if val.lower() in ('true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '0', '1'):
            bool_count += 1

        # Check integer
        try:
            int(val)
            int_count += 1
            continue
        except ValueError:
            pass

        # Check float
        try:
            float(val)
            float_count += 1
            continue
        except ValueError:
            pass

    total = len([v for v in values if v.strip()])
    if total == 0:
        return 'string'

    # If >80% are numeric, consider it numeric
    if int_count / total > 0.8:
        return 'integer'
    if float_count / total > 0.8:
        return 'float'
    if bool_count / total > 0.8:
        return 'boolean'

    return 'string'


def _is_numeric_column(file_path: Path, column_index: int, sample_size: int = 100) -> bool:
    """Check if a column contains numeric data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            values = []
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break
                if column_index < len(row):
                    values.append(row[column_index])

            data_type = _infer_data_type(values)
            return data_type in ('integer', 'float')
    except:
        return False


def _format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format data as a markdown table."""
    if not rows:
        return "No data to display."

    # Build table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        # Ensure row has same length as headers
        padded_row = row + [""] * (len(headers) - len(row))
        padded_row = padded_row[:len(headers)]
        # Escape pipe characters in cell values
        escaped_row = [str(cell).replace("|", "\\|") for cell in padded_row]
        lines.append("| " + " | ".join(escaped_row) + " |")

    return "\n".join(lines)


def _truncate_if_needed(content: str, message_prefix: str = "") -> str:
    """Truncate content if it exceeds CHARACTER_LIMIT."""
    if len(content) <= CHARACTER_LIMIT:
        return content

    truncation_msg = f"\n\n---\n**⚠️ Output truncated** (exceeded {CHARACTER_LIMIT:,} character limit). {message_prefix}"
    available_space = CHARACTER_LIMIT - len(truncation_msg)

    return content[:available_space] + truncation_msg


def _run_bash_command(command: List[str], input_data: Optional[str] = None) -> tuple[bool, str]:
    """
    Run a bash command safely.

    Returns:
        tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return False, f"Command failed: {result.stderr}"

        return True, result.stdout

    except subprocess.TimeoutExpired:
        return False, "Error: Command timed out after 30 seconds."
    except Exception as e:
        return False, f"Error: Command execution failed: {str(e)}"


# Tool Implementations

@mcp.tool(
    name="csv_inspect_structure",
    annotations={
        "title": "Inspect CSV Structure",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_inspect_structure(params: InspectStructureInput) -> str:
    """
    Get comprehensive overview of CSV file structure and contents.

    This tool reads CSV file headers, infers data types, counts rows and columns,
    identifies missing values, and provides a sample of the data. Use this as the
    first step to understand what's in a CSV file before performing analysis or
    creating visualizations.

    Args:
        params (InspectStructureInput): Validated input containing:
            - file_path (str): Path to CSV file
            - sample_rows (int): Number of sample rows to display (default: 5)

    Returns:
        str: Markdown-formatted output containing:
            - File path and size
            - Total rows and columns
            - Column names with inferred data types
            - Sample rows from beginning of file
            - Missing value counts per column

    Examples:
        - Use when: "What's in this CSV file?"
        - Use when: "Show me the structure of data.csv"
        - Use when: "What columns are available in sales.csv?"

    Error Handling:
        - Returns clear error if file not found with suggestion to check path
        - Returns error if file is not readable
        - Returns error if CSV is malformed or empty
    """
    # Validate file
    valid, error_msg, file_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Read headers
    success, error_msg, headers = _read_csv_headers(file_path)
    if not success:
        return error_msg

    try:
        # Get file size
        file_size = file_path.stat().st_size
        file_size_str = f"{file_size:,} bytes"
        if file_size > 1024 * 1024:
            file_size_str = f"{file_size / (1024*1024):.2f} MB"
        elif file_size > 1024:
            file_size_str = f"{file_size / 1024:.2f} KB"

        # Count total rows (using wc -l for efficiency)
        success, output = _run_bash_command(['wc', '-l', str(file_path)])
        total_rows = 0
        if success:
            total_rows = int(output.split()[0]) - 1  # Subtract header row

        # Read sample rows and infer types
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            sample_rows = []
            column_values: Dict[int, List[str]] = {i: [] for i in range(len(headers))}

            for i, row in enumerate(reader):
                if i < params.sample_rows:
                    sample_rows.append(row)

                # Collect values for type inference (up to 100 rows)
                if i < 100:
                    for col_idx, value in enumerate(row):
                        if col_idx < len(headers):
                            column_values[col_idx].append(value)

        # Infer data types
        data_types = []
        missing_counts = []
        for col_idx in range(len(headers)):
            values = column_values[col_idx]
            data_types.append(_infer_data_type(values))
            missing = sum(1 for v in values if not v.strip())
            missing_counts.append(missing)

        # Build markdown output
        lines = [
            f"# CSV File Structure: {file_path.name}",
            "",
            f"**Path:** `{file_path}`",
            f"**Size:** {file_size_str}",
            f"**Rows:** {total_rows:,}",
            f"**Columns:** {len(headers)}",
            "",
            "## Column Information",
            ""
        ]

        # Column table
        col_table = [
            "| # | Column Name | Data Type | Missing Values |",
            "|---|-------------|-----------|----------------|"
        ]
        for i, (header, dtype, missing) in enumerate(zip(headers, data_types, missing_counts), 1):
            missing_str = f"{missing}" if missing > 0 else "-"
            col_table.append(f"| {i} | {header} | {dtype} | {missing_str} |")

        lines.extend(col_table)
        lines.append("")

        # Sample data
        if sample_rows:
            lines.append(f"## Sample Data (first {len(sample_rows)} rows)")
            lines.append("")
            lines.append(_format_markdown_table(headers, sample_rows))
        else:
            lines.append("## Sample Data")
            lines.append("")
            lines.append("*No data rows found in file.*")

        result = "\n".join(lines)
        return _truncate_if_needed(
            result,
            "Use 'csv_preview_rows' with specific row ranges to see more data."
        )

    except Exception as e:
        return f"Error: Failed to inspect CSV file: {str(e)}"


@mcp.tool(
    name="csv_get_column_info",
    annotations={
        "title": "Get Column Information",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_get_column_info(params: ColumnInfoInput) -> str:
    """
    Get detailed information about a specific column including unique values and statistics.

    This tool analyzes a single column in depth, providing unique value counts, frequency
    distributions, and statistical measures (for numeric columns). Use this to understand
    the distribution and range of values in a column before filtering or visualizing.

    Args:
        params (ColumnInfoInput): Validated input containing:
            - file_path (str): Path to CSV file
            - column_name (str): Exact column name to analyze
            - max_unique (int): Maximum unique values to display (default: 100)

    Returns:
        str: Markdown-formatted output containing:
            - Column name and data type
            - Total unique value count
            - Top N unique values with frequencies
            - For numeric columns: min, max, mean, median, std deviation
            - Missing value count

    Examples:
        - Use when: "What are the possible values for 'country' column?"
        - Use when: "Show me statistics for the 'price' column"
        - Use when: "How many unique categories are in 'product_type'?"

    Error Handling:
        - Returns error if file not found
        - Returns error if column name doesn't exist (lists available columns)
        - Handles numeric conversion errors gracefully
    """
    # Validate file
    valid, error_msg, file_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Read headers and find column
    success, error_msg, headers = _read_csv_headers(file_path)
    if not success:
        return error_msg

    success, error_msg, col_idx = _get_column_index(headers, params.column_name)
    if not success:
        return error_msg

    try:
        # Extract column values using cut and count unique with sort | uniq -c
        # This is more efficient than loading entire file into Python
        success, output = _run_bash_command([
            'bash', '-c',
            f"cut -d',' -f{col_idx + 1} '{file_path}' | tail -n +2 | sort | uniq -c | sort -rn"
        ])

        if not success:
            return f"Error: Failed to analyze column: {output}"

        # Parse the output (format: "  count value")
        lines = output.strip().split('\n')
        unique_values = []
        total_count = 0
        missing_count = 0

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                count = int(parts[0])
                value = parts[1]
                if not value.strip():
                    missing_count = count
                else:
                    unique_values.append((value, count))
                total_count += count
            elif len(parts) == 1:
                # Empty value
                missing_count = int(parts[0])
                total_count += missing_count

        # Check if numeric column
        is_numeric = _is_numeric_column(file_path, col_idx)

        # Build output
        lines = [
            f"# Column Analysis: {params.column_name}",
            "",
            f"**File:** `{file_path.name}`",
            f"**Total Values:** {total_count:,}",
            f"**Unique Values:** {len(unique_values):,}",
            f"**Missing Values:** {missing_count}",
            "",
        ]

        # Compute statistics for numeric columns
        if is_numeric and unique_values:
            numeric_values = []
            for value, count in unique_values:
                try:
                    num_val = float(value)
                    numeric_values.extend([num_val] * count)
                except:
                    pass

            if numeric_values:
                lines.append("## Statistics")
                lines.append("")
                lines.append(f"- **Min:** {min(numeric_values)}")
                lines.append(f"- **Max:** {max(numeric_values)}")
                lines.append(f"- **Mean:** {statistics.mean(numeric_values):.2f}")
                lines.append(f"- **Median:** {statistics.median(numeric_values):.2f}")
                if len(numeric_values) > 1:
                    lines.append(f"- **Std Dev:** {statistics.stdev(numeric_values):.2f}")
                lines.append("")

        # Show top unique values
        display_count = min(params.max_unique, len(unique_values))
        lines.append(f"## Top {display_count} Values by Frequency")
        lines.append("")

        if unique_values:
            # Create frequency table
            freq_table = [
                "| Value | Count | Percentage |",
                "|-------|-------|------------|"
            ]
            for value, count in unique_values[:display_count]:
                pct = (count / total_count * 100) if total_count > 0 else 0
                escaped_value = str(value).replace("|", "\\|")
                freq_table.append(f"| {escaped_value} | {count:,} | {pct:.1f}% |")

            lines.extend(freq_table)

            if len(unique_values) > display_count:
                lines.append("")
                lines.append(f"*Showing top {display_count} of {len(unique_values):,} unique values. Use 'max_unique' parameter to see more.*")
        else:
            lines.append("*No non-empty values found.*")

        result = "\n".join(lines)
        return _truncate_if_needed(
            result,
            f"Increase 'max_unique' or use 'csv_get_unique_values' to see more values."
        )

    except Exception as e:
        return f"Error: Failed to analyze column: {str(e)}"


@mcp.tool(
    name="csv_get_unique_values",
    annotations={
        "title": "Get Unique Values",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_get_unique_values(params: UniqueValuesInput) -> str:
    """
    List all unique values in a column (optimized for categorical data).

    This tool efficiently extracts and lists unique values from a column, useful for
    understanding categorical variables, identifying distinct categories, or preparing
    filter conditions for downstream processing.

    Args:
        params (UniqueValuesInput): Validated input containing:
            - file_path (str): Path to CSV file
            - column_name (str): Exact column name
            - limit (Optional[int]): Maximum number of values to return

    Returns:
        str: Markdown-formatted list of unique values, sorted alphabetically

    Examples:
        - Use when: "List all countries in the dataset"
        - Use when: "What are the unique product categories?"
        - Use when: "Show me all possible status values"

    Error Handling:
        - Returns error if file not found
        - Returns error if column doesn't exist (lists available columns)
    """
    # Validate file
    valid, error_msg, file_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Read headers and find column
    success, error_msg, headers = _read_csv_headers(file_path)
    if not success:
        return error_msg

    success, error_msg, col_idx = _get_column_index(headers, params.column_name)
    if not success:
        return error_msg

    try:
        # Use bash utilities to extract unique values efficiently
        cmd = f"cut -d',' -f{col_idx + 1} '{file_path}' | tail -n +2 | sort -u"
        success, output = _run_bash_command(['bash', '-c', cmd])

        if not success:
            return f"Error: Failed to extract unique values: {output}"

        unique_values = [line for line in output.strip().split('\n') if line.strip()]
        total_count = len(unique_values)

        # Apply limit if specified
        if params.limit and params.limit < total_count:
            unique_values = unique_values[:params.limit]
            truncated = True
        else:
            truncated = False

        # Build output
        lines = [
            f"# Unique Values: {params.column_name}",
            "",
            f"**File:** `{file_path.name}`",
            f"**Total Unique Values:** {total_count:,}",
            ""
        ]

        if truncated:
            lines.append(f"**Showing:** First {params.limit} values (sorted alphabetically)")
            lines.append("")

        lines.append("## Values")
        lines.append("")

        if unique_values:
            for value in unique_values:
                escaped_value = value.replace("|", "\\|").replace("\n", " ")
                lines.append(f"- `{escaped_value}`")

            if truncated:
                lines.append("")
                lines.append(f"*Showing {len(unique_values)} of {total_count:,} unique values. Increase 'limit' parameter to see more.*")
        else:
            lines.append("*No non-empty values found.*")

        result = "\n".join(lines)
        return _truncate_if_needed(result, "Use 'limit' parameter to control output size.")

    except Exception as e:
        return f"Error: Failed to get unique values: {str(e)}"


@mcp.tool(
    name="csv_get_summary_statistics",
    annotations={
        "title": "Get Summary Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_get_summary_statistics(params: SummaryStatisticsInput) -> str:
    """
    Get statistical summary of numeric columns in CSV file.

    This tool computes comprehensive statistics (count, min, max, mean, median, std dev)
    for numeric columns. Use this to quickly understand the distribution and range of
    numeric data before creating plots or performing analysis.

    Args:
        params (SummaryStatisticsInput): Validated input containing:
            - file_path (str): Path to CSV file
            - columns (Optional[List[str]]): Specific columns to analyze (if not provided, analyzes all numeric columns)

    Returns:
        str: Markdown-formatted table with statistics for each numeric column

    Examples:
        - Use when: "Give me statistics for all numeric columns"
        - Use when: "Show me summary stats for price and quantity columns"
        - Use when: "What's the range of values in the 'age' column?"

    Error Handling:
        - Returns error if file not found
        - Returns error if specified columns don't exist
        - Skips non-numeric columns with a note
    """
    # Validate file
    valid, error_msg, file_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Read headers
    success, error_msg, headers = _read_csv_headers(file_path)
    if not success:
        return error_msg

    try:
        # Determine which columns to analyze
        if params.columns:
            # Validate specified columns exist
            columns_to_analyze = []
            for col_name in params.columns:
                success, error_msg, col_idx = _get_column_index(headers, col_name)
                if not success:
                    return error_msg
                columns_to_analyze.append((col_name, col_idx))
        else:
            # Auto-detect numeric columns
            columns_to_analyze = []
            for col_idx, col_name in enumerate(headers):
                if _is_numeric_column(file_path, col_idx):
                    columns_to_analyze.append((col_name, col_idx))

        if not columns_to_analyze:
            return "No numeric columns found to analyze. Try specifying column names explicitly or use 'csv_inspect_structure' to see available columns."

        # Read and compute statistics
        stats_results = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            # Collect all values
            column_values: Dict[str, List[float]] = {col_name: [] for col_name, _ in columns_to_analyze}

            for row in reader:
                for col_name, col_idx in columns_to_analyze:
                    if col_idx < len(row):
                        value = row[col_idx].strip()
                        if value:
                            try:
                                column_values[col_name].append(float(value))
                            except ValueError:
                                pass  # Skip non-numeric values

        # Compute statistics
        for col_name, col_idx in columns_to_analyze:
            values = column_values[col_name]

            if not values:
                stats_results.append({
                    'column': col_name,
                    'count': 0,
                    'min': 'N/A',
                    'max': 'N/A',
                    'mean': 'N/A',
                    'median': 'N/A',
                    'std_dev': 'N/A'
                })
                continue

            stats_results.append({
                'column': col_name,
                'count': len(values),
                'min': f"{min(values):.2f}",
                'max': f"{max(values):.2f}",
                'mean': f"{statistics.mean(values):.2f}",
                'median': f"{statistics.median(values):.2f}",
                'std_dev': f"{statistics.stdev(values):.2f}" if len(values) > 1 else "N/A"
            })

        # Build output
        lines = [
            f"# Summary Statistics: {file_path.name}",
            "",
            f"**Columns Analyzed:** {len(stats_results)}",
            "",
            "## Statistics Table",
            ""
        ]

        # Create statistics table
        table = [
            "| Column | Count | Min | Max | Mean | Median | Std Dev |",
            "|--------|-------|-----|-----|------|--------|---------|"
        ]

        for stats in stats_results:
            table.append(
                f"| {stats['column']} | {stats['count']} | {stats['min']} | "
                f"{stats['max']} | {stats['mean']} | {stats['median']} | {stats['std_dev']} |"
            )

        lines.extend(table)

        result = "\n".join(lines)
        return _truncate_if_needed(result)

    except Exception as e:
        return f"Error: Failed to compute statistics: {str(e)}"


@mcp.tool(
    name="csv_filter_and_transform",
    annotations={
        "title": "Filter and Transform CSV",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_filter_and_transform(params: FilterTransformInput) -> str:
    """
    Create a filtered and/or transformed CSV file for downstream processing.

    This tool creates a new CSV file by selecting specific columns, filtering rows,
    and optionally sorting the data. The output file can be passed to other tools
    (e.g., R/ggplot MCP) for visualization. This is especially useful when you need
    to prepare a subset of data for plotting or further analysis.

    Args:
        params (FilterTransformInput): Validated input containing:
            - file_path (str): Input CSV file path
            - output_path (str): Output CSV file path
            - select_columns (Optional[List[str]]): Columns to include (default: all)
            - filter_conditions (Optional[List[str]]): Filters like "column=value" or "column!=value"
            - sort_by (Optional[str]): Column name to sort by

    Returns:
        str: Markdown-formatted output with:
            - Output file path
            - Number of rows in output
            - Preview of first few rows
            - Summary of transformations applied

    Examples:
        - Use when: "Create a CSV with only US sales for visualization"
        - Use when: "Filter to active users and save to filtered_users.csv"
        - Use when: "Extract price and quantity columns sorted by date"

    Error Handling:
        - Returns error if input file not found
        - Returns error if output path is invalid
        - Returns error if columns don't exist
        - Returns error if filter syntax is invalid
    """
    # Validate input file
    valid, error_msg, input_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Resolve output path
    output_path = _resolve_path(params.output_path)

    # Check if output directory exists
    if not output_path.parent.exists():
        return f"Error: Output directory '{output_path.parent}' does not exist."

    # Read headers
    success, error_msg, headers = _read_csv_headers(input_path)
    if not success:
        return error_msg

    try:
        # Determine columns to select
        if params.select_columns:
            # Validate all columns exist
            selected_indices = []
            for col_name in params.select_columns:
                success, error_msg, col_idx = _get_column_index(headers, col_name)
                if not success:
                    return error_msg
                selected_indices.append(col_idx)
            output_headers = params.select_columns
        else:
            selected_indices = list(range(len(headers)))
            output_headers = headers

        # Parse filter conditions
        filters = []
        if params.filter_conditions:
            for condition in params.filter_conditions:
                if '!=' in condition:
                    col_name, value = condition.split('!=', 1)
                    col_name = col_name.strip()
                    value = value.strip()
                    success, error_msg, col_idx = _get_column_index(headers, col_name)
                    if not success:
                        return error_msg
                    filters.append(('!=', col_idx, value))
                elif '=' in condition:
                    col_name, value = condition.split('=', 1)
                    col_name = col_name.strip()
                    value = value.strip()
                    success, error_msg, col_idx = _get_column_index(headers, col_name)
                    if not success:
                        return error_msg
                    filters.append(('=', col_idx, value))
                else:
                    return f"Error: Invalid filter condition '{condition}'. Use format 'column=value' or 'column!=value'."

        # Determine sort column
        sort_col_idx = None
        if params.sort_by:
            success, error_msg, sort_col_idx = _get_column_index(headers, params.sort_by)
            if not success:
                return error_msg

        # Read, filter, and write
        filtered_rows = []

        with open(input_path, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            next(reader)  # Skip header

            for row in reader:
                # Apply filters
                passes_filter = True
                for op, col_idx, value in filters:
                    if col_idx >= len(row):
                        passes_filter = False
                        break
                    cell_value = row[col_idx].strip()

                    if op == '=':
                        if cell_value != value:
                            passes_filter = False
                            break
                    elif op == '!=':
                        if cell_value == value:
                            passes_filter = False
                            break

                if not passes_filter:
                    continue

                # Select columns
                selected_row = [row[i] if i < len(row) else "" for i in selected_indices]
                filtered_rows.append(selected_row)

        # Sort if requested
        if sort_col_idx is not None and params.sort_by in output_headers:
            output_sort_idx = output_headers.index(params.sort_by)
            try:
                # Try numeric sort first
                filtered_rows.sort(key=lambda r: float(r[output_sort_idx]) if r[output_sort_idx].strip() else float('inf'))
            except (ValueError, IndexError):
                # Fall back to string sort
                filtered_rows.sort(key=lambda r: r[output_sort_idx] if output_sort_idx < len(r) else "")

        # Write output file
        with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(output_headers)
            writer.writerows(filtered_rows)

        # Build output
        lines = [
            f"# CSV Transform Complete",
            "",
            f"**Input:** `{input_path.name}`",
            f"**Output:** `{output_path}`",
            f"**Output Rows:** {len(filtered_rows):,}",
            "",
            "## Transformations Applied",
            ""
        ]

        if params.select_columns:
            lines.append(f"- **Selected Columns:** {', '.join(params.select_columns)}")
        else:
            lines.append(f"- **Selected Columns:** All ({len(headers)} columns)")

        if params.filter_conditions:
            lines.append(f"- **Filters:** {len(params.filter_conditions)}")
            for condition in params.filter_conditions:
                lines.append(f"  - `{condition}`")
        else:
            lines.append("- **Filters:** None")

        if params.sort_by:
            lines.append(f"- **Sorted By:** {params.sort_by}")
        else:
            lines.append("- **Sorted By:** None")

        # Show preview
        if filtered_rows:
            preview_count = min(5, len(filtered_rows))
            lines.append("")
            lines.append(f"## Preview (first {preview_count} rows)")
            lines.append("")
            lines.append(_format_markdown_table(output_headers, filtered_rows[:preview_count]))
        else:
            lines.append("")
            lines.append("## Preview")
            lines.append("")
            lines.append("*No rows matched the filter conditions.*")

        result = "\n".join(lines)
        return _truncate_if_needed(result)

    except Exception as e:
        return f"Error: Failed to transform CSV: {str(e)}"


@mcp.tool(
    name="csv_preview_rows",
    annotations={
        "title": "Preview CSV Rows",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def csv_preview_rows(params: PreviewRowsInput) -> str:
    """
    Preview specific rows from a CSV file.

    This tool displays a specific range of rows from the CSV file, useful for
    examining data at different positions (not just the beginning). Use this to
    spot-check data quality or inspect specific sections of the file.

    Args:
        params (PreviewRowsInput): Validated input containing:
            - file_path (str): Path to CSV file
            - start_row (int): Starting row number (1-indexed, excluding header)
            - num_rows (int): Number of rows to display (default: 10)

    Returns:
        str: Markdown-formatted table showing the requested rows

    Examples:
        - Use when: "Show me rows 100-110 of the data"
        - Use when: "Preview the middle section of the file"
        - Use when: "Let me see rows near the end"

    Error Handling:
        - Returns error if file not found
        - Handles cases where requested rows exceed file length
        - Returns error if file is empty
    """
    # Validate file
    valid, error_msg, file_path = _validate_csv_file(params.file_path)
    if not valid:
        return error_msg

    # Read headers
    success, error_msg, headers = _read_csv_headers(file_path)
    if not success:
        return error_msg

    try:
        # Use head and tail to efficiently extract specific rows
        # tail -n +N starts from line N (1-indexed)
        # head -n M takes first M lines
        start_line = params.start_row + 1  # +1 because of header
        end_line = start_line + params.num_rows - 1

        # Use sed to extract specific line range
        cmd = f"sed -n '{start_line},{end_line}p' '{file_path}'"
        success, output = _run_bash_command(['bash', '-c', cmd])

        if not success:
            return f"Error: Failed to read rows: {output}"

        if not output.strip():
            return f"No rows found at position {params.start_row}. File may have fewer rows than requested."

        # Parse rows
        rows = []
        for line in output.strip().split('\n'):
            reader = csv.reader([line])
            rows.extend(reader)

        # Build output
        lines = [
            f"# CSV Preview: {file_path.name}",
            "",
            f"**Rows:** {params.start_row} to {params.start_row + len(rows) - 1}",
            "",
            _format_markdown_table(headers, rows)
        ]

        result = "\n".join(lines)
        return _truncate_if_needed(result)

    except Exception as e:
        return f"Error: Failed to preview rows: {str(e)}"


# Entry point
if __name__ == "__main__":
    mcp.run()
