# 100m Freestyle Swimming: Swimmer Profiling & Performance Modelling

Quantitative sports analytics study of elite 100m Freestyle swimming performances using World Aquatics All-Time Top 200 records (1980-2022) and Olympic Games results (1988-2020). All spectral decompositions (SVD / PCA), rank-2 matrix completion, and regression estimators under collinearity (OLS, Moore-Penrose, Ridge with LOO cross-validation, and 2D Fisher confidence ellipsoids) are implemented directly from linear algebra primitives in `NumPy` and `SciPy`.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Raw CSV datasets (top_perf_categories.csv, Results_1912to2020.csv)
│   └── processed/            # Cleaned tables and multi-event swimmer matrices
├── notebooks/                # Exploratory Jupyter notebook
├── notes/                    # Step-by-step technical summaries (Phases 1 to 3)
├── paper/                    # LaTeX report (paper.tex, paper.pdf) and generated figures/
├── src/
│   └── sports_analytics_l3/
│       ├── data_cleaning.py  # Phase 1: parsing, alias resolution, table construction
│       ├── spectral_pca.py   # Phase 2: gender standardization, reduced SVD/PCA, biplot
│       └── modelling.py      # Phase 3: collinearity diagnostics, OLS/MP/Ridge, LOO CV, Fisher ellipses
└── study_guide/              # Interactive HTML/JS companion guide
```

## Setup & Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Phase 1: Clean raw data and build processed CSV tables
uv run python src/sports_analytics_l3/data_cleaning.py

# Phase 2: Run SVD / PCA spectral analysis and generate Phase 2 figures
uv run python src/sports_analytics_l3/spectral_pca.py

# Phase 3: Run collinearity diagnostics, OLS/Ridge LOO CV, Fisher ellipses, and Phase 3 figures
uv run python src/sports_analytics_l3/modelling.py
```

To compile the LaTeX report (`paper/paper.pdf`):

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex && latexmk -c paper.tex
```

## Author

Paul-Emile Razafindrakoto
