"""
Spectral analysis, SVD, and PCA built from scratch in NumPy.

Studies the latent structure around the 100m Freestyle on two matrices:
1. Complete Olympic finalists matrix (1988-2020, no NaNs) as an unbiased
   spectral reference for SVD, Eckart-Young-Mirsky errors, and conditioning.
2. Multi-event sprint swimmer matrix completed by iterative rank-2 SVD to
   map individual swimmer profiles on a 2D PCA biplot.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

OLYMPIC_EVENTS: tuple[str, ...] = (
    "50m Freestyle",
    "100m Freestyle",
    "200m Freestyle",
    "400m Freestyle",
    "100m Butterfly",
    "200m Individual Medley",
)

SPRINT_CORE_EVENTS: tuple[str, ...] = (
    "50m Freestyle",
    "100m Freestyle",
    "200m Freestyle",
    "100m Butterfly",
)

SHORT_EVENT_LABELS: dict[str, str] = {
    "50m Freestyle": "50m Free",
    "100m Freestyle": "100m Free",
    "200m Freestyle": "200m Free",
    "400m Freestyle": "400m Free",
    "100m Butterfly": "100m Fly",
    "200m Individual Medley": "200m IM",
}


@dataclass(frozen=True)
class SVDPcaResult:
    """Outputs of a reduced SVD / PCA decomposition on a standardized matrix."""

    X_std: np.ndarray
    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    eckart_young_errors: np.ndarray
    cond_X: float
    cond_XtX: float


def standardize_matrix(
    X: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Center and scale columns of X (globally or within each group, e.g. gender).

    Uses nanmean/nanstd so sparse swimmer matrices keep their NaN mask intact.
    """
    X_arr = np.asarray(X, dtype=float)
    X_out = np.empty_like(X_arr)

    if groups is None:
        masks = [np.ones(X_arr.shape[0], dtype=bool)]
    else:
        group_arr = np.asarray(groups)
        masks = [group_arr == g for g in np.unique(group_arr)]

    for mask in masks:
        sub = X_arr[mask]
        mean = np.nanmean(sub, axis=0)
        std = np.nanstd(sub, axis=0, ddof=0)
        # Guard against zero-variance columns so division stays well-defined.
        std = np.where(std == 0.0, 1.0, std)
        X_out[mask] = (sub - mean) / std
    return X_out


def fit_svd_pca(X_std: np.ndarray) -> SVDPcaResult:
    """Compute reduced SVD (X = U Sigma V^T), PCA scores, and spectral metrics."""
    U, s, Vt = np.linalg.svd(X_std, full_matrices=False)

    # Fix singular vector sign ambiguity so faster times (negative z) map to PC1 > 0.
    if np.sum(Vt[0]) > 0:
        U[:, 0] *= -1.0
        Vt[0, :] *= -1.0

    if Vt.shape[0] > 1 and Vt[1, -1] < 0:
        U[:, 1] *= -1.0
        Vt[1, :] *= -1.0

    scores = U @ np.diag(s)
    sq_s = s**2
    total_energy = float(np.sum(sq_s))
    explained_ratio = sq_s / total_energy

    # Relative rank-k Frobenius error rho_k from the Eckart-Young-Mirsky theorem.
    eckart_young = np.array(
        [np.sum(sq_s[k:]) / total_energy for k in range(1, len(s) + 1)],
        dtype=float,
    )

    cond_X = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    cond_XtX = cond_X**2

    return SVDPcaResult(
        X_std=X_std,
        U=U,
        singular_values=s,
        Vt=Vt,
        scores=scores,
        explained_variance_ratio=explained_ratio,
        eckart_young_errors=eckart_young,
        cond_X=cond_X,
        cond_XtX=cond_XtX,
    )


def verify_svd_identities(pca: SVDPcaResult, tol: float = 1e-10) -> dict[str, float]:
    """Verify SVD energy conservation, Eckart-Young error, and orthogonality."""
    X = pca.X_std
    s = pca.singular_values
    Vt = pca.Vt
    Z = pca.scores

    frob_X_sq = float(np.linalg.norm(X, ord="fro") ** 2)
    sum_sq_s = float(np.sum(s**2))
    frob_diff = abs(frob_X_sq - sum_sq_s)

    k = min(2, len(s))
    X_hat_k = (pca.U[:, :k] * s[:k]) @ Vt[:k, :]
    rho_direct = float(np.linalg.norm(X - X_hat_k, ord="fro") ** 2 / frob_X_sq)
    rho_spectral = float(pca.eckart_young_errors[k - 1])
    eckart_diff = abs(rho_direct - rho_spectral)

    ortho_err = float(np.max(np.abs(Vt @ Vt.T - np.eye(len(s)))))
    decorr_err = float(np.max(np.abs(Z.T @ Z - np.diag(s**2))))

    assert frob_diff < tol, f"Frobenius check failed: {frob_diff}"
    assert eckart_diff < tol, f"Eckart-Young check failed: {eckart_diff}"
    assert ortho_err < tol, f"Orthogonality check failed: {ortho_err}"
    assert decorr_err < tol * max(1.0, frob_X_sq), f"Score decorrelation check failed: {decorr_err}"

    return {
        "frobenius_sq": frob_X_sq,
        "sum_sigma_sq": sum_sq_s,
        "frobenius_abs_diff": frob_diff,
        "eckart_young_diff": eckart_diff,
        "ortho_max_err": ortho_err,
        "decorr_max_err": decorr_err,
    }


def impute_missing_svd(
    X_with_nan: np.ndarray,
    rank: int = 2,
    max_iter: int = 300,
    tol: float = 1e-7,
) -> np.ndarray:
    """Fill missing entries iteratively via truncated rank-k SVD reconstruction."""
    obs = ~np.isnan(X_with_nan)
    row_means = np.nanmean(X_with_nan, axis=1, keepdims=True)
    X_imp = np.where(obs, X_with_nan, row_means)

    # Alternate rank-k projection with clamping observed entries until convergence.
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(X_imp, full_matrices=False)
        X_hat = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
        X_next = np.where(obs, X_with_nan, X_hat)
        if np.linalg.norm(X_next - X_imp) / max(1.0, np.linalg.norm(X_imp)) < tol:
            X_imp = X_next
            break
        X_imp = X_next

    return standardize_matrix(X_imp)


# --- Data loading and figures ---


def load_olympic_event_matrix(
    csv_path: Path = PROCESSED_DIR / "olympic_results_cleaned.csv",
    events: tuple[str, ...] = OLYMPIC_EVENTS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build the complete (year x gender) x events matrix of Olympic averages."""
    grouped: dict[tuple[int, str], dict[str, list[float]]] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ev = row["event"]
            if ev not in events:
                continue
            key = (int(row["year"]), row["gender"])
            grouped.setdefault(key, {}).setdefault(ev, []).append(float(row["swim_time_s"]))

    keys = sorted(grouped.keys(), key=lambda k: (k[1], k[0]))
    X = np.array(
        [[float(np.mean(grouped[k][ev])) for ev in events] for k in keys],
        dtype=float,
    )
    labels = [f"{year} ({'M' if g == 'Men' else 'W'})" for year, g in keys]
    genders = np.array([g for _, g in keys])
    return X, labels, genders


def load_swimmer_sprint_matrix(
    csv_path: Path = PROCESSED_DIR / "multi_event_matrix.csv",
    events: tuple[str, ...] = SPRINT_CORE_EVENTS,
    min_events: int = 2,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load swimmers with at least min_events recorded across the four sprint events."""
    rows_kept: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if sum(1 for ev in events if row[ev] != "") >= min_events:
                rows_kept.append(row)

    X = np.full((len(rows_kept), len(events)), np.nan, dtype=float)
    for i, row in enumerate(rows_kept):
        for j, ev in enumerate(events):
            if row[ev] != "":
                X[i, j] = float(row[ev])

    names = [row["athlete_name"] for row in rows_kept]
    genders = np.array([row["gender"] for row in rows_kept])
    return X, names, genders


def plot_scree_and_eckart_young(
    pca_olympic: SVDPcaResult,
    out_path: Path = FIGURES_DIR / "phase2_scree_eckart_young.png",
) -> Path:
    """Plot the singular value spectrum and Eckart-Young-Mirsky rank-k error."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranks = np.arange(1, len(pca_olympic.singular_values) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.patch.set_facecolor("white")

    ax1.bar(ranks, pca_olympic.singular_values, color="#0077B6", alpha=0.88, width=0.58)
    ax1.plot(ranks, pca_olympic.singular_values, color="#0A2540", marker="o", linewidth=1.6)
    for k, s_val, var_r in zip(
        ranks, pca_olympic.singular_values, pca_olympic.explained_variance_ratio
    ):
        ax1.text(
            k,
            s_val + 0.25,
            f"{var_r * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#0A2540",
        )
    ax1.set_title("Singular Value Spectrum (Olympics 1988–2020)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Component index k")
    ax1.set_ylabel(r"Singular value $\sigma_k$")
    ax1.set_xticks(ranks)
    ax1.set_ylim(0, max(pca_olympic.singular_values) * 1.16)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2.plot(
        ranks,
        pca_olympic.eckart_young_errors * 100.0,
        color="#00B4D8",
        marker="s",
        linewidth=2.2,
        markersize=6,
    )
    for k, rho in zip(ranks, pca_olympic.eckart_young_errors):
        ax2.text(k, rho * 100.0 + 0.22, f"{rho * 100:.2f}%", ha="center", fontsize=8.5)
    ax2.set_title(r"Eckart-Young Approximation Error ($\rho_k$)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Approximation rank k")
    ax2.set_ylabel("Residual relative error (%)")
    ax2.set_xticks(ranks)
    ax2.set_ylim(-0.3, max(pca_olympic.eckart_young_errors) * 118.0)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_sprint_biplot(
    pca_swimmers: SVDPcaResult,
    names: list[str],
    genders: np.ndarray,
    events: tuple[str, ...] = SPRINT_CORE_EVENTS,
    out_path: Path = FIGURES_DIR / "phase2_biplot_sprint.png",
) -> Path:
    """Plot the (PC1, PC2) PCA biplot of multi-event sprint swimmers and event vectors."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Z = pca_swimmers.scores[:, :2]
    V2 = pca_swimmers.Vt[:2, :]
    var1, var2 = pca_swimmers.explained_variance_ratio[:2] * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=150)
    fig.patch.set_facecolor("white")

    for gender, color, label in [
        ("Men", "#0077B6", "Men"),
        ("Women", "#EE334E", "Women"),
    ]:
        mask = genders == gender
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=color,
            alpha=0.65,
            s=38,
            edgecolors="white",
            linewidth=0.5,
            label=label,
        )

    arrow_scale = 0.85 * min(np.max(np.abs(Z[:, 0])), np.max(np.abs(Z[:, 1])))
    for j, ev in enumerate(events):
        vx, vy = V2[0, j] * arrow_scale, V2[1, j] * arrow_scale
        ax.annotate(
            "",
            xy=(vx, vy),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "color": "#0A2540", "lw": 1.8},
        )
        ax.text(
            vx * 1.12,
            vy * 1.12,
            SHORT_EVENT_LABELS.get(ev, ev),
            fontsize=9.5,
            fontweight="bold",
            color="#0A2540",
            ha="center",
            va="center",
        )

    featured = {
        "Caeleb Dressel": (8, 5),
        "David Popovici": (8, 4),
        "Michael Phelps": (8, 6),
        "Sarah Sjoestroem": (8, 8),
        "Emma Mckeon": (8, -12),
        "Matthew Biondi": (-8, -12),
        "Catherine Plewinski": (8, 6),
        "Kyle Chalmers": (8, -10),
    }
    name_to_idx = {name: i for i, name in enumerate(names)}
    for athlete, (dx, dy) in featured.items():
        if athlete not in name_to_idx:
            continue
        idx = name_to_idx[athlete]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=65, facecolors="none", edgecolors="#0A2540", lw=1.2)
        ax.annotate(
            athlete,
            xy=(Z[idx, 0], Z[idx, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.2,
            color="#1E293B",
            ha="left" if dx >= 0 else "right",
        )

    ax.axhline(0.0, color="#94A3B8", lw=0.8, ls="--")
    ax.axvline(0.0, color="#94A3B8", lw=0.8, ls="--")
    ax.set_xlim(float(np.min(Z[:, 0])) - 0.5, float(np.max(Z[:, 0])) + 0.9)
    ax.set_title("PCA Biplot: Sprint Profiles of Multi-Event Swimmers", fontsize=11.5, fontweight="bold")
    ax.set_xlabel(f"PC1: Overall sprint level ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2: 100m Free vs. 200m Free contrast ({var2:.1f}% variance)")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_phase2_pipeline() -> dict[str, object]:
    """Run the full SVD/PCA spectral analysis and export both figures."""
    X_oly_raw, oly_labels, oly_genders = load_olympic_event_matrix()
    X_oly_std = standardize_matrix(X_oly_raw, groups=oly_genders)
    pca_olympic = fit_svd_pca(X_oly_std)
    oly_checks = verify_svd_identities(pca_olympic)

    X_swim_raw, swim_names, swim_genders = load_swimmer_sprint_matrix()
    X_swim_std = standardize_matrix(X_swim_raw, groups=swim_genders)
    X_swim_imp = impute_missing_svd(X_swim_std, rank=2)
    pca_swimmers = fit_svd_pca(X_swim_imp)
    swim_checks = verify_svd_identities(pca_swimmers)

    fig_scree = plot_scree_and_eckart_young(pca_olympic)
    fig_biplot = plot_sprint_biplot(pca_swimmers, swim_names, swim_genders)

    return {
        "olympic_labels": oly_labels,
        "pca_olympic": pca_olympic,
        "olympic_checks": oly_checks,
        "swimmer_names": swim_names,
        "pca_swimmers": pca_swimmers,
        "swimmer_checks": swim_checks,
        "figures": [fig_scree, fig_biplot],
    }


if __name__ == "__main__":
    # 1. Unit tests on synthetic data (standardization, NaN groups, SVD identities)
    rng = np.random.default_rng(42)
    A_test = rng.normal(loc=50.0, scale=3.0, size=(20, 4))
    A_std = standardize_matrix(A_test)
    assert np.allclose(A_std.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(A_std.std(axis=0, ddof=0), 1.0, atol=1e-12)

    G_test = np.array(["A"] * 10 + ["B"] * 10)
    A_nan = A_test.copy()
    A_nan[0, 0] = np.nan
    A_nan_std = standardize_matrix(A_nan, groups=G_test)
    assert np.isnan(A_nan_std[0, 0])
    assert np.allclose(np.nanmean(A_nan_std[:10], axis=0), 0.0, atol=1e-12)

    pca_test = fit_svd_pca(A_std)
    verify_svd_identities(pca_test)

    # 2. Full run on real project matrices
    results = run_phase2_pipeline()
    pca_oly: SVDPcaResult = results["pca_olympic"]  # type: ignore[assignment]
    pca_swim: SVDPcaResult = results["pca_swimmers"]  # type: ignore[assignment]

    assert pca_oly.X_std.shape == (18, 6)
    assert np.isclose(pca_oly.explained_variance_ratio.sum(), 1.0, atol=1e-12)
    assert pca_oly.eckart_young_errors[-1] < 1e-12
    assert pca_swim.X_std.shape == (128, 4)
    for fig_path in results["figures"]:  # type: ignore[attr-defined]
        assert Path(fig_path).exists() and Path(fig_path).stat().st_size > 0

    print("=== SVD / PCA spectral analysis: all checks passed ===")
    print(f"Olympics 1988-2020 (m={pca_oly.X_std.shape[0]}, n={pca_oly.X_std.shape[1]}):")
    print("  Singular values  :", np.round(pca_oly.singular_values, 3))
    print("  Explained var (%) :", np.round(pca_oly.explained_variance_ratio * 100, 2))
    print("  Eckart-Young (%)  :", np.round(pca_oly.eckart_young_errors * 100, 3))
    print(f"  kappa_2(X) = {pca_oly.cond_X:.2f} | kappa_2(X^T X) = {pca_oly.cond_XtX:.2f}")
    print(f"\nMulti-event sprint swimmers (m={pca_swim.X_std.shape[0]}, n={pca_swim.X_std.shape[1]}):")
    print("  Singular values  :", np.round(pca_swim.singular_values, 3))
    print("  Explained var (%) :", np.round(pca_swim.explained_variance_ratio * 100, 2))
    print("Generated figures:", [str(p.relative_to(PROJECT_ROOT)) for p in results["figures"]])  # type: ignore[attr-defined]
