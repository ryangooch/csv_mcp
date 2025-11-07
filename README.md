# CSV MCP Server

A Model Context Protocol (MCP) server that provides simple utilities for understanding and operating on CSV files. This server uses bash utilities wrapped in Python to efficiently inspect, analyze, and transform CSV files, making it easy for LLMs to understand CSV contents and prepare data for visualization tools like ggplot.

## Features

### Tools

1. **`csv_inspect_structure`** - Get comprehensive overview of CSV file structure
   - Column names with inferred data types
   - Row and column counts
   - Sample data from beginning of file
   - Missing value counts

2. **`csv_get_column_info`** - Deep analysis of specific columns
   - Unique value counts and frequencies
   - Statistical measures (min, max, mean, median, std dev) for numeric columns
   - Data type detection

3. **`csv_get_unique_values`** - List unique values in a column
   - Efficiently extracts unique values using bash utilities
   - Useful for categorical variables

4. **`csv_get_summary_statistics`** - Statistical summary of numeric columns
   - Comprehensive statistics across all or selected numeric columns
   - Count, min, max, mean, median, standard deviation

5. **`csv_filter_and_transform`** - Create filtered/transformed CSV files
   - Select specific columns
   - Filter rows based on conditions
   - Sort by column values
   - Output to new file for downstream processing (e.g., ggplot via R MCP)

6. **`csv_preview_rows`** - Preview specific row ranges
   - View data at any position in the file
   - Useful for spot-checking data quality

## Installation

This server uses `uv` for Python package management:

```bash
cd csv_mcp
uv sync
```

## Usage

### Running the Server

The server uses stdio transport by default:

```bash
uv run python server.py
```

### Using with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "csv": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/csv_mcp/server.py"]
    }
  }
}
```

### Example Workflow

1. **Inspect a CSV file:**
   ```
   Use csv_inspect_structure with file_path="sales_data.csv"
   ```

2. **Analyze specific columns:**
   ```
   Use csv_get_column_info with file_path="sales_data.csv" and column_name="region"
   ```

3. **Create filtered dataset for visualization:**
   ```
   Use csv_filter_and_transform with:
   - file_path="sales_data.csv"
   - output_path="us_sales.csv"
   - filter_conditions=["region=US"]
   - select_columns=["date", "revenue", "product"]
   ```

4. **Pass to R/ggplot MCP for visualization:**
   ```
   Now use the R MCP server with us_sales.csv to create visualizations
   ```

## Design Philosophy

- **Simplicity:** Uses standard bash utilities (cut, sort, uniq, sed, head, tail) for efficiency
- **Focus:** Designed to help LLMs understand CSV contents, not replace full data analysis tools
- **Integration:** Prepares data for handoff to other tools (especially R/ggplot for visualization)
- **Markdown Output:** All responses are human and LLM readable markdown

## Requirements

- Python 3.10+
- Standard Unix utilities (available on macOS and Linux)
- MCP Python SDK
- Pydantic v2

## Security

- All file paths are validated and resolved
- Only reads files (except csv_filter_and_transform which creates new files)
- Commands are executed safely with timeouts
- No arbitrary command execution

## License

MIT
