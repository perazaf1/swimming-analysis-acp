# Swimming Analysis — PCA & Statistical Modelling

A data analysis project studying elite swimming performances (100m Freestyle and related events) using Principal Component Analysis (PCA / SVD) and linear statistical modelling.

## Project Structure

```text
.
├── data/
│   ├── raw/            # Raw datasets (top_perf_categories.csv, Results_1912to2020.csv)
│   └── processed/      # Cleaned tables and multi-event matrices
├── notebooks/          # Exploratory Jupyter notebooks
├── src/
│   └── sports_analytics_l3/  # Python scripts for cleaning and PCA
└── paper/              # LaTeX report
```

## Setup & Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run data cleaning
uv run python src/sports_analytics_l3/data_cleaning.py
```

## Author

Paul-Emile Razafindrakoto
