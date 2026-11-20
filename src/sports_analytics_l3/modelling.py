"""
Phase 3: 100m Freestyle performance modelling under collinearity.

Implements collinearity diagnostics (condition numbers, VIF), linear regression
estimators (OLS, Moore-Penrose pseudoinverse, Ridge regularization), analytical
and explicit Leave-One-Out cross-validation, and Fisher information 95%
confidence intervals directly from NumPy and SciPy linear algebra primitives.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def compute_fisher_confidence_intervals(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Compute Fisher information matrix, inverse covariance, standard errors, and (1 - alpha) CIs."""
    res = fit_ols(X, y)
    n, p = X.shape
    df_resid = n - p
    sigma2 = float(np.sum(res.residuals**2) / df_resid)
    XtX = X.T @ X
    fisher_info = XtX / sigma2
    cov_beta = sigma2 * np.linalg.inv(XtX)
    se_beta = np.sqrt(np.diag(cov_beta))
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df_resid))
    half_widths = t_crit * se_beta
    ci_low = res.beta - half_widths
    ci_high = res.beta + half_widths

    return {
        "beta": res.beta,
        "fisher_info": fisher_info,
        "cov_beta": cov_beta,
        "se_beta": se_beta,
        "t_crit": t_crit,
        "half_widths": half_widths,
        "ci_low": ci_low,
        "ci_high": ci_high,
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


def plot_fisher_confidence_intervals(
    datasets: tuple[RegressionDataset, RegressionDataset],
    out_path: Path = FIGURES_DIR / "phase3_fisher_ci.png",
) -> Path:
    """Plot 95% confidence intervals from inverse Fisher information vs. Ridge weights."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes, datasets):
        ci_info = compute_fisher_confidence_intervals(ds.X_std, ds.y_std, alpha=0.05)
        beta_ols = np.asarray(ci_info["beta"], dtype=float)
        hw = np.asarray(ci_info["half_widths"], dtype=float)
        diag = diagnose_collinearity(ds.X_std)
        vifs = np.asarray(diag["vif"], dtype=float)

        cv_info = select_ridge_lambda(ds.X_std, ds.y_std)
        best_lam = float(cv_info["best_lam"])  # type: ignore[arg-type]
        beta_r_best = fit_ridge(ds.X_std, ds.y_std, lam=best_lam).beta
        beta_r2 = fit_ridge(ds.X_std, ds.y_std, lam=2.0).beta

        p = len(ds.feature_names)
        y_pos = np.arange(p)[::-1]
        labels = [
            f"{SHORT_EVENT_LABELS.get(f, f)}\n(VIF={vifs[j]:.1f})"
            for j, f in enumerate(ds.feature_names)
        ]

        ax.axvline(0.0, color="#94A3B8", lw=1.1, ls="--")

        # 1. 95% OLS / MLE confidence intervals from inverse Fisher information
        ax.errorbar(
            beta_ols,
            y_pos + 0.10,
            xerr=hw,
            fmt="o",
            color="#0077B6",
            ecolor="#0077B6",
            elinewidth=2.0,
            capsize=4.5,
            markersize=6.0,
            label="OLS / MLE ± 95% Fisher CI",
            zorder=4,
        )

        # 2. Optimal Ridge and Ridge(lambda=2.0) weights
        ax.scatter(
            beta_r_best,
            y_pos - 0.08,
            color="#0A2540",
            marker="D",
            s=48,
            zorder=5,
            label=rf"Ridge ($\lambda^*={best_lam:.3g}$)",
        )
        ax.scatter(
            beta_r2,
            y_pos - 0.22,
            color="#EE334E",
            marker="^",
            s=52,
            zorder=5,
            label=r"Ridge ($\lambda=2.0$)",
        )

        kind = "Olympic 5-Event" if ds.name.startswith("olympic") else "Individual Sprint"
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_ylim(-0.95, p - 0.35)
        ax.set_title(
            f"{kind} (n={ds.X_std.shape[0]}, df={ci_info['df_resid']})",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel(r"Standardized weight $\beta_j$ and 95% confidence interval")
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            fontsize=7.8,
            loc="lower right",
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
    fig2 = plot_fisher_confidence_intervals((ds_oly_5, ds_sprint))
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

    print("\n=== Step 4: Fisher Information and 95% Confidence Intervals ===")
    for ds in (ds_oly_5, ds_oly_2, ds_sprint):
        ci = compute_fisher_confidence_intervals(ds.X_std, ds.y_std, alpha=0.05)
        n, p = ds.X_std.shape
        diag = diagnose_collinearity(ds.X_std)
        fisher_info = np.asarray(ci["fisher_info"], dtype=float)
        cov_beta = np.asarray(ci["cov_beta"], dtype=float)
        se_beta = np.asarray(ci["se_beta"], dtype=float)
        hw = np.asarray(ci["half_widths"], dtype=float)

        # Verify I_n(beta) * Sigma_beta == I_p and SE(beta_j) == sigma_hat * sqrt(VIF_j / n)
        assert np.allclose(fisher_info @ cov_beta, np.eye(p), atol=1e-10)
        assert np.allclose(
            se_beta,
            np.sqrt(float(ci["sigma2"]) * np.asarray(diag["vif"], dtype=float) / n),
            atol=1e-10,
        )
        print(
            f"[{ds.name}] beta={np.round(ci['beta'], 3)} "
            f"| SE={np.round(se_beta, 3)} | 95% CI half-width=±{np.round(hw, 3)}"
        )

    fig1 = plot_ridge_path_and_cv(ds_oly_5)
    assert fig1.exists() and fig1.stat().st_size > 0
    print("\nGenerated figure:", fig1)

    fig2 = plot_fisher_confidence_intervals((ds_oly_5, ds_sprint))
    assert fig2.exists() and fig2.stat().st_size > 0
    print("Generated figure:", fig2)

    fig3 = plot_predictions_in_seconds(ds_sprint, lam=0.0)
    assert fig3.exists() and fig3.stat().st_size > 0
    print("Generated figure:", fig3)
