# 100m Freestyle Swimming: Profiling & Modelling

Analysis of elite 100m Freestyle performances from World Aquatics Top 200 records (1980-2022) and Olympic Games results (1988-2020), using SVD/PCA and Ridge regression coded directly in `NumPy` and `SciPy`.

The full report is available in [`paper/paper.pdf`](paper/paper.pdf).

## Run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

uv run python src/sports_analytics_l3/data_cleaning.py
uv run python src/sports_analytics_l3/spectral_pca.py
uv run python src/sports_analytics_l3/modelling.py
```

## Author

Paul-Emile Razafindrakoto
