"""
Main package for the 100m Freestyle swimming analysis and statistical modelling project.
"""

from __future__ import annotations

from sports_analytics_l3.data_cleaning import run_phase1_pipeline
from sports_analytics_l3.modelling import run_phase3_pipeline
from sports_analytics_l3.spectral_pca import run_phase2_pipeline


def main() -> None:
    """Execute the full three-phase 100m Freestyle analysis pipeline."""
    print("[Phase 1] Cleaning raw CSV datasets and exporting processed tables...")
    p1 = run_phase1_pipeline()
    print("  Exported tables:", ", ".join(f"{name}.csv ({len(df)} rows)" for name, df in p1.items()))

    print("[Phase 2] Running SVD / PCA spectral analysis and generating biplots...")
    p2 = run_phase2_pipeline()
    print("  Generated figures:", ", ".join(str(p.name) for p in p2["figures"]))  # type: ignore[attr-defined]

    print("[Phase 3] Running collinearity diagnostics, Ridge LOO CV, and Fisher 95% CIs...")
    p3 = run_phase3_pipeline()
    print("  Generated figures:", ", ".join(str(p.name) for p in p3["figures"]))  # type: ignore[attr-defined]
    print("All three phases completed successfully.")


if __name__ == "__main__":
    main()
