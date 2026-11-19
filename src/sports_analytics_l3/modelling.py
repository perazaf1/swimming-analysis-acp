"""
Phase 3: 100m Freestyle performance modelling under collinearity.

Implements collinearity diagnostics (condition numbers, VIF), linear regression
estimators (OLS, Moore-Penrose pseudoinverse, Ridge regularization), analytical
and explicit Leave-One-Out cross-validation, and 2D Fisher confidence ellipsoids
directly from NumPy and SciPy linear algebra primitives.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from sports_analytics_l3.spectral_pca import (
    FIGURES_DIR,
    OLYMPIC_EVENTS,
    PROCESSED_DIR,
    SHORT_EVENT_LABELS,
    load_olympic_event_matrix,
    standardize_matrix,
)


@dataclass(frozen=True)
class RegressionDataset:
    """Complete gender-standardized regression dataset (observed rows only, no SVD-imputed values)."""

    name: str
    X_raw: np.ndarray  # shape (n, p) in seconds
    y_raw: np.ndarray  # shape (n,) in seconds
    X_std: np.ndarray  # shape (n, p) standardized within gender
    y_std: np.ndarray  # shape (n,) standardized within gender
    feature_names: tuple[str, ...]
    sample_labels: list[str]
    genders: np.ndarray


def load_individual_dataset(
    predictors: tuple[str, ...],
    target: str = "100m Freestyle",
    csv_path: Path = PROCESSED_DIR / "multi_event_matrix.csv",
) -> RegressionDataset:
    """Load complete individual swimmer rows observed across target and predictors."""
    rows_kept: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row[target] != "" and all(row[col] != "" for col in predictors):
                rows_kept.append(row)

    X_raw = np.array(
        [[float(row[col]) for col in predictors] for row in rows_kept],
        dtype=float,
    )
    y_raw = np.array([float(row[target]) for row in rows_kept], dtype=float)
    sample_labels = [row["athlete_name"] for row in rows_kept]
    genders = np.array([row["gender"] for row in rows_kept])

    X_std = standardize_matrix(X_raw, groups=genders)
    y_std = standardize_matrix(y_raw[:, None], groups=genders).ravel()

    return RegressionDataset(
        name=f"individual ({', '.join(predictors)} -> {target})",
        X_raw=X_raw,
        y_raw=y_raw,
        X_std=X_std,
        y_std=y_std,
        feature_names=tuple(predictors),
        sample_labels=sample_labels,
        genders=genders,
    )


def load_olympic_regression_dataset(
    predictors: tuple[str, ...] | None = None,
    target: str = "100m Freestyle",
) -> RegressionDataset:
    """Build a complete regression dataset from the 1988-2020 Olympic finals matrix."""
    if predictors is None:
        predictors = tuple(ev for ev in OLYMPIC_EVENTS if ev != target)
    X_all, labels, genders = load_olympic_event_matrix()
    target_idx = OLYMPIC_EVENTS.index(target)
    pred_idx = [OLYMPIC_EVENTS.index(ev) for ev in predictors]
    X_raw = X_all[:, pred_idx]
    y_raw = X_all[:, target_idx]
    X_std = standardize_matrix(X_raw, groups=genders)
    y_std = standardize_matrix(y_raw[:, None], groups=genders).ravel()

    return RegressionDataset(
        name=f"olympic ({', '.join(predictors)} -> {target})",
        X_raw=X_raw,
        y_raw=y_raw,
        X_std=X_std,
        y_std=y_std,
        feature_names=tuple(predictors),
        sample_labels=labels,
        genders=genders,
    )


def diagnose_collinearity(X: np.ndarray) -> dict[str, object]:
    """Compute singular values, L2 condition numbers, and Variance Inflation Factors."""
    s = np.linalg.svd(X, compute_uv=False)
    cond_X = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    cond_XtX = cond_X**2

    p = X.shape[1]
    if p == 1:
        vif = np.array([1.0])
    else:
        corr = np.corrcoef(X, rowvar=False)
        vif = np.diag(np.linalg.inv(corr))

    return {
        "singular_values": s,
        "cond_X": cond_X,
        "cond_XtX": cond_XtX,
        "vif": vif,
    }


@dataclass(frozen=True)
class RegressionResult:
    """Fitted linear regression weights and in-sample metrics."""

    method: str
    beta: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    rmse: float
    r2: float


def _build_regression_result(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
) -> RegressionResult:
    y_pred = X @ beta
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else 0.0
    return RegressionResult(
        method=method,
        beta=beta,
        y_pred=y_pred,
        residuals=residuals,
        rmse=rmse,
        r2=r2,
    )


def fit_ols(X: np.ndarray, y: np.ndarray) -> RegressionResult:
    """Fit Ordinary Least Squares via normal equations (X^T X) beta = X^T y."""
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    return _build_regression_result("OLS", X, y, beta)


def fit_moore_penrose(
    X: np.ndarray,
    y: np.ndarray,
    rcond: float = 1e-12,
) -> RegressionResult:
    """Fit minimum-norm least squares via the SVD Moore-Penrose pseudoinverse."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    beta = Vt.T @ (s_inv * (U.T @ y))
    return _build_regression_result("Moore-Penrose", X, y, beta)


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> RegressionResult:
    """Fit L2-regularized Ridge regression (X^T X + lam I_p) beta = X^T y."""
    p = X.shape[1]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    return _build_regression_result(f"Ridge(lam={lam})", X, y, beta)


def leave_one_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Compute explicit Leave-One-Out cross-validation RMSE and error vector."""
    n = X.shape[0]
    errors = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.arange(n) != i
        res = fit_ridge(X[mask], y[mask], lam=lam)
        errors[i] = y[i] - float(X[i] @ res.beta)
    rmse_cv = float(np.sqrt(np.mean(errors**2)))
    return rmse_cv, errors


def select_ridge_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate Ridge along a logarithmic lambda grid and select LOO-optimal lambda*."""
    if lambdas is None:
        lambdas = np.logspace(-3, 2, 100)

    betas_list: list[np.ndarray] = []
    train_list: list[float] = []
    cv_list: list[float] = []

    for lam in lambdas:
        res = fit_ridge(X, y, float(lam))
        rmse_cv, _ = leave_one_out_cv(X, y, lam=float(lam))
        betas_list.append(res.beta)
        train_list.append(res.rmse)
        cv_list.append(rmse_cv)

    betas = np.array(betas_list, dtype=float)
    train_rmses = np.array(train_list, dtype=float)
    cv_rmses = np.array(cv_list, dtype=float)

    best_idx = int(np.argmin(cv_rmses))
    best_lam = float(lambdas[best_idx])
    best_cv_rmse = float(cv_rmses[best_idx])

    return {
        "lambdas": lambdas,
        "betas": betas,
        "train_rmses": train_rmses,
        "cv_rmses": cv_rmses,
        "best_lam": best_lam,
        "best_cv_rmse": best_cv_rmse,
    }


def errors_std_to_seconds(
    errors_std: np.ndarray,
    y_raw: np.ndarray,
    genders: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Convert standardized errors back into race seconds using within-gender target std."""
    errors_sec = np.empty_like(errors_std, dtype=float)
    for g in np.unique(genders):
        mask = genders == g
        sigma_y = float(np.std(y_raw[mask], ddof=0))
        errors_sec[mask] = errors_std[mask] * sigma_y
    rmse_sec = float(np.sqrt(np.mean(errors_sec**2)))
    return rmse_sec, errors_sec


def compute_fisher_ellipse_2d(
    X: np.ndarray,
    y: np.ndarray,
    idx: tuple[int, int] = (0, 1),
    alpha: float = 0.05,
    n_points: int = 200,
) -> dict[str, object]:
    """Compute the joint (1 - alpha) 2D Fisher confidence ellipsoid and marginal t-intervals."""
    res = fit_ols(X, y)
    n, p = X.shape
    df_resid = n - p
    i, j = idx
    sigma2 = float(np.sum(res.residuals**2) / df_resid)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    center = res.beta[[i, j]]
    cov_2d = cov_beta[np.ix_([i, j], [i, j])]
    f_crit = float(f_dist.ppf(1.0 - alpha, 2, df_resid))
    radius_sq = 2.0 * f_crit
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    axes = eigvecs * np.sqrt(radius_sq * eigvals)
    ellipse_pts = center[:, None] + axes @ unit_circle  # shape (2, n_points)
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df_resid))
    marginal_half_widths = t_crit * np.sqrt(np.diag(cov_2d))

    return {
        "center": center,
        "cov_2d": cov_2d,
        "radius_sq": radius_sq,
        "ellipse_pts": ellipse_pts,
        "marginal_half_widths": marginal_half_widths,
        "sigma2": sigma2,
        "df_resid": df_resid,
    }


def plot_ridge_path_and_cv(
    ds: RegressionDataset,
    out_path: Path = FIGURES_DIR / "phase3_ridge_path_cv.png",
) -> Path:
    """Plot Ridge regularization path (left) and LOO CV error curve (right)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv_info = select_ridge_lambda(ds.X_std, ds.y_std)
    ols_cv, _ = leave_one_out_cv(ds.X_std, ds.y_std, lam=0.0)

    lambdas = np.asarray(cv_info["lambdas"], dtype=float)
    betas = np.asarray(cv_info["betas"], dtype=float)
    train_rmses = np.asarray(cv_info["train_rmses"], dtype=float)
    cv_rmses = np.asarray(cv_info["cv_rmses"], dtype=float)
    best_lam = float(cv_info["best_lam"])  # type: ignore[arg-type]
    best_cv = float(cv_info["best_cv_rmse"])  # type: ignore[arg-type]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=150)
    fig.patch.set_facecolor("white")

    # Left panel: Ridge coefficient trace
    colors = ["#0077B6", "#00B4D8", "#EE334E", "#0A2540", "#F59E0B"]
    for j, feat in enumerate(ds.feature_names):
        label = SHORT_EVENT_LABELS.get(feat, feat)
        ax1.semilogx(lambdas, betas[:, j], lw=2.0, color=colors[j % len(colors)], label=label)

    ax1.axhline(0.0, color="#94A3B8", lw=0.9, ls="--")
    ax1.axvline(best_lam, color="#0A2540", lw=1.3, ls=":", label=rf"$\lambda^* = {best_lam:.3f}$")
    ax1.set_title("Ridge Trace (Standardized Coefficients)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel(r"Regularization parameter $\lambda$")
    ax1.set_ylabel(r"Estimated weight $\hat{\beta}_j(\lambda)$")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.2)

    # Right panel: LOO CV vs in-sample RMSE across lambda
    ax2.semilogx(lambdas, cv_rmses, color="#0077B6", lw=2.2, label="LOO CV RMSE")
    ax2.semilogx(lambdas, train_rmses, color="#94A3B8", lw=1.6, ls="--", label="Train RMSE")
    ax2.axhline(ols_cv, color="#EE334E", lw=1.3, ls="-.", label=f"OLS LOO ({ols_cv:.3f})")
    ax2.scatter([best_lam], [best_cv], color="#0A2540", s=55, zorder=5, label=f"Min CV ({best_cv:.3f})")
    ax2.axvline(best_lam, color="#0A2540", lw=1.1, ls=":")

    ax2.set_title("Leave-One-Out Cross-Validation Error", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel(r"Regularization parameter $\lambda$")
    ax2.set_ylabel("RMSE (standardized units)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fisher_ellipses(
    datasets: tuple[RegressionDataset, RegressionDataset],
    out_path: Path = FIGURES_DIR / "phase3_fisher_ellipses.png",
) -> Path:
    """Plot 95% 2D Fisher confidence ellipsoids, marginal boxes, and Ridge paths."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, datasets):
        ell = compute_fisher_ellipse_2d(ds.X_std, ds.y_std, idx=(0, 1), alpha=0.05)
        center = np.asarray(ell["center"], dtype=float)
        pts = np.asarray(ell["ellipse_pts"], dtype=float)
        hw = np.asarray(ell["marginal_half_widths"], dtype=float)
        cov_2d = np.asarray(ell["cov_2d"], dtype=float)

        eigvals = np.linalg.eigvalsh(cov_2d)
        ratio = float(np.sqrt(eigvals[1] / eigvals[0]))

        cv_info = select_ridge_lambda(ds.X_std, ds.y_std)
        betas_path = np.asarray(cv_info["betas"], dtype=float)
        best_lam = float(cv_info["best_lam"])  # type: ignore[arg-type]
        beta_r_best = fit_ridge(ds.X_std, ds.y_std, lam=best_lam).beta
        beta_r2 = fit_ridge(ds.X_std, ds.y_std, lam=2.0).beta

        # 1. Joint 95% Fisher confidence ellipse
        ax.fill(pts[0], pts[1], color="#0077B6", alpha=0.14)
        ax.plot(pts[0], pts[1], color="#0077B6", lw=2.0, label="95% Fisher ellipse")

        # 2. Marginal 95% Student t-interval box
        rect_x = [
            center[0] - hw[0],
            center[0] + hw[0],
            center[0] + hw[0],
            center[0] - hw[0],
            center[0] - hw[0],
        ]
        rect_y = [
            center[1] - hw[1],
            center[1] - hw[1],
            center[1] + hw[1],
            center[1] + hw[1],
            center[1] - hw[1],
        ]
        ax.plot(rect_x, rect_y, color="#64748B", lw=1.1, ls="--", label="95% marginal t-CI")

        # 3. Ridge regularization trajectory and key estimators
        ax.plot(betas_path[:, 0], betas_path[:, 1], color="#0A2540", lw=1.4, ls=":", label="Ridge path")
        ax.scatter(center[0], center[1], color="#EE334E", s=60, zorder=5, label="OLS")
        ax.scatter(
            beta_r_best[0],
            beta_r_best[1],
            color="#0A2540",
            marker="D",
            s=50,
            zorder=6,
            label=rf"Ridge ($\lambda^*={best_lam:.2f}$)",
        )
        ax.scatter(
            beta_r2[0],
            beta_r2[1],
            color="#F59E0B",
            marker="^",
            s=55,
            zorder=6,
            label=r"Ridge ($\lambda=2.0$)",
        )

        ax.axhline(0.0, color="#CBD5E1", lw=0.9)
        ax.axvline(0.0, color="#CBD5E1", lw=0.9)
        ax.set_aspect("equal", adjustable="datalim")

        f0 = SHORT_EVENT_LABELS.get(ds.feature_names[0], ds.feature_names[0])
        f1 = SHORT_EVENT_LABELS.get(ds.feature_names[1], ds.feature_names[1])
        kind = "Olympic" if ds.name.startswith("olympic") else "Individual"
        ax.set_title(
            f"{kind} (n={ds.X_std.shape[0]}, axis ratio={ratio:.2f}x)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel(rf"Weight on {f0} ($\beta_1$)")
        ax.set_ylabel(rf"Weight on {f1} ($\beta_2$)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            fontsize=7.6,
            loc="upper right",
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_predictions_in_seconds(
    ds: RegressionDataset,
    lam: float = 0.0,
    out_path: Path = FIGURES_DIR / "phase3_predictions_seconds.png",
) -> Path:
    """Plot LOO predicted vs actual 100m times (left) and per-swimmer residuals (right) in seconds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, errs_std = leave_one_out_cv(ds.X_std, ds.y_std, lam=lam)
    rmse_sec, errs_sec = errors_std_to_seconds(errs_std, ds.y_raw, ds.genders)
    y_pred_sec = ds.y_raw - errs_sec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=150)
    fig.patch.set_facecolor("white")

    # Left panel: LOO predicted vs actual times
    for gender, color in [("Men", "#0077B6"), ("Women", "#EE334E")]:
        mask = ds.genders == gender
        ax1.scatter(
            ds.y_raw[mask],
            y_pred_sec[mask],
            c=color,
            s=55,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.7,
            label=gender,
        )

    lims = [
        min(ds.y_raw.min(), y_pred_sec.min()) - 0.5,
        max(ds.y_raw.max(), y_pred_sec.max()) + 0.5,
    ]
    ax1.plot(lims, lims, color="#0A2540", lw=1.4, ls="--", label=r"Perfect fit ($y = \hat{y}$)")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(
        f"LOO 100m Predictions (n={len(ds.y_raw)}, RMSE={rmse_sec:.3f} s)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_xlabel("Actual 100m Freestyle (s)")
    ax1.set_ylabel("Predicted 100m Freestyle (s)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)

    # Right panel: Per-swimmer residuals in seconds
    order = np.argsort(errs_sec)
    sorted_errs = errs_sec[order]
    sorted_names = [ds.sample_labels[k] for k in order]
    sorted_colors = ["#0077B6" if ds.genders[k] == "Men" else "#EE334E" for k in order]

    y_pos = np.arange(len(sorted_names))
    ax2.barh(y_pos, sorted_errs, color=sorted_colors, alpha=0.85, height=0.65)
    ax2.axvline(0.0, color="#0A2540", lw=1.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=8.0)
    ax2.set_title("LOO Residuals by Swimmer (Actual - Predicted)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Residual (s)  [< 0: faster than predicted]")
    ax2.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_phase3_pipeline() -> dict[str, object]:
    """Run the full Phase 3 regression modelling analysis and export all three figures."""
    ds_sprint = load_individual_dataset(("50m Freestyle", "100m Butterfly"))
    ds_50 = load_individual_dataset(("50m Freestyle",))
    ds_oly_5 = load_olympic_regression_dataset()
    ds_oly_2 = load_olympic_regression_dataset(predictors=("50m Freestyle", "200m Freestyle"))

    fig1 = plot_ridge_path_and_cv(ds_oly_5)
    fig2 = plot_fisher_ellipses((ds_oly_2, ds_sprint))
    fig3 = plot_predictions_in_seconds(ds_sprint, lam=0.0)

    return {
        "datasets": {
            "individual_sprint": ds_sprint,
            "individual_50m": ds_50,
            "olympic_5": ds_oly_5,
            "olympic_2": ds_oly_2,
        },
        "figures": [fig1, fig2, fig3],
    }


if __name__ == "__main__":
    # 1. Individual datasets (bivariate sprint + univariate baseline)
    ds_sprint = load_individual_dataset(("50m Freestyle", "100m Butterfly"))
    assert ds_sprint.X_std.shape == (14, 2)
    assert ds_sprint.y_std.shape == (14,)

    ds_50 = load_individual_dataset(("50m Freestyle",))
    assert ds_50.X_std.shape == (56, 1)

    # 2. Olympic datasets (5 complete predictors + highly collinear 2D pair)
    ds_oly_5 = load_olympic_regression_dataset()
    assert ds_oly_5.X_std.shape == (18, 5)
    assert ds_oly_5.y_std.shape == (18,)

    ds_oly_2 = load_olympic_regression_dataset(predictors=("50m Freestyle", "200m Freestyle"))
    assert ds_oly_2.X_std.shape == (18, 2)

    # 3. Common checks: zero NaNs, within-gender standardization, and collinearity
    for ds in (ds_sprint, ds_50, ds_oly_5, ds_oly_2):
        assert not np.isnan(ds.X_std).any()
        assert not np.isnan(ds.y_std).any()
        for g in np.unique(ds.genders):
            mask = ds.genders == g
            assert np.allclose(ds.X_std[mask].mean(axis=0), 0.0, atol=1e-12)
            assert np.allclose(ds.X_std[mask].std(axis=0, ddof=0), 1.0, atol=1e-12)
            assert np.allclose(ds.y_std[mask].mean(), 0.0, atol=1e-12)

        diag = diagnose_collinearity(ds.X_std)
        print(f"{ds.name} -> X={ds.X_std.shape}, y={ds.y_std.shape}")
        print(f"  Men: {(ds.genders == 'Men').sum()} | Women: {(ds.genders == 'Women').sum()}")
        print(
            f"  cond(X)={diag['cond_X']:.2f} | cond(X^T X)={diag['cond_XtX']:.2f} "
            f"| VIF={np.round(diag['vif'], 2)}"
        )

    print("\n=== Step 2: OLS vs Moore-Penrose vs Ridge ===")
    for ds in (ds_oly_5, ds_oly_2, ds_sprint):
        res_ols = fit_ols(ds.X_std, ds.y_std)
        res_mp = fit_moore_penrose(ds.X_std, ds.y_std)
        res_r0 = fit_ridge(ds.X_std, ds.y_std, lam=0.0)
        res_r2 = fit_ridge(ds.X_std, ds.y_std, lam=2.0)

        # Algebraic identity: full column rank => OLS == Moore-Penrose == Ridge(0)
        assert np.allclose(res_ols.beta, res_mp.beta, atol=1e-10)
        assert np.allclose(res_ols.beta, res_r0.beta, atol=1e-10)

        # Verify spectral SVD form of Ridge: V diag(s / (s^2 + lam)) U^T y
        U, s, Vt = np.linalg.svd(ds.X_std, full_matrices=False)
        beta_ridge_svd = Vt.T @ ((s / (s**2 + 2.0)) * (U.T @ ds.y_std))
        assert np.allclose(res_r2.beta, beta_ridge_svd, atol=1e-10)

        print(f"\n[{ds.name}]")
        for res in (res_ols, res_mp, res_r2):
            print(
                f"  {res.method:<15} | beta={np.round(res.beta, 3)} "
                f"| ||beta||={np.linalg.norm(res.beta):.3f} "
                f"| R2={res.r2:.4f} | RMSE_std={res.rmse:.4f}"
            )

    print("\n=== Step 3: LOO Cross-Validation, lambda* selection, and RMSE in seconds ===")
    for ds in (ds_oly_5, ds_oly_2, ds_sprint, ds_50):
        # Verify analytical PRESS / hat-matrix identity H_lam
        test_lam = 1.5
        rmse_loo_loop, errs_loo_loop = leave_one_out_cv(ds.X_std, ds.y_std, lam=test_lam)
        res_test = fit_ridge(ds.X_std, ds.y_std, lam=test_lam)
        p = ds.X_std.shape[1]
        H_lam = ds.X_std @ np.linalg.solve(
            ds.X_std.T @ ds.X_std + test_lam * np.eye(p), ds.X_std.T
        )
        errs_loo_hat = res_test.residuals / (1.0 - np.diag(H_lam))
        assert np.allclose(errs_loo_loop, errs_loo_hat, atol=1e-10)
        assert np.isclose(rmse_loo_loop, np.sqrt(np.mean(errs_loo_hat**2)), atol=1e-12)

        # LOO comparison: OLS (lam=0) vs Ridge (optimal lam*)
        ols_cv_std, ols_errs_std = leave_one_out_cv(ds.X_std, ds.y_std, lam=0.0)
        ols_cv_sec, _ = errors_std_to_seconds(ols_errs_std, ds.y_raw, ds.genders)

        cv_info = select_ridge_lambda(ds.X_std, ds.y_std)
        best_lam = float(cv_info["best_lam"])  # type: ignore[arg-type]
        ridge_cv_std, ridge_errs_std = leave_one_out_cv(ds.X_std, ds.y_std, lam=best_lam)
        ridge_cv_sec, _ = errors_std_to_seconds(ridge_errs_std, ds.y_raw, ds.genders)
        res_best = fit_ridge(ds.X_std, ds.y_std, lam=best_lam)

        print(f"\n[{ds.name}]")
        print(f"  OLS (LOO)            | RMSE_cv={ols_cv_std:.4f} std ({ols_cv_sec:.3f} s)")
        print(
            f"  Ridge(lam*={best_lam:.4f}) | RMSE_cv={ridge_cv_std:.4f} std ({ridge_cv_sec:.3f} s) "
            f"| beta={np.round(res_best.beta, 3)}"
        )

    print("\n=== Step 4: 2D Fisher Confidence Ellipsoids ===")
    for ds in (ds_oly_2, ds_sprint):
        ell = compute_fisher_ellipse_2d(ds.X_std, ds.y_std, idx=(0, 1), alpha=0.05)
        center = ell["center"]
        cov_2d = ell["cov_2d"]
        pts = ell["ellipse_pts"]
        # Verify contour points satisfy (x - c)^T Sigma^-1 (x - c) == radius_sq
        diff = pts - center[:, None]
        inv_cov = np.linalg.inv(cov_2d)
        quad_forms = np.sum(diff * (inv_cov @ diff), axis=0)
        assert np.allclose(quad_forms, ell["radius_sq"], atol=1e-10)

        eigvals = np.linalg.eigvalsh(cov_2d)
        semi_axes = np.sqrt(ell["radius_sq"] * eigvals)
        print(
            f"[{ds.name}] center={np.round(center, 3)} "
            f"| semi_axes={np.round(semi_axes, 3)} (ratio={semi_axes[1] / semi_axes[0]:.2f}x)"
        )

    fig1 = plot_ridge_path_and_cv(ds_oly_5)
    assert fig1.exists() and fig1.stat().st_size > 0
    print("\nGenerated figure:", fig1)

    fig2 = plot_fisher_ellipses((ds_oly_2, ds_sprint))
    assert fig2.exists() and fig2.stat().st_size > 0
    print("Generated figure:", fig2)

    fig3 = plot_predictions_in_seconds(ds_sprint, lam=0.0)
    assert fig3.exists() and fig3.stat().st_size > 0
    print("Generated figure:", fig3)
