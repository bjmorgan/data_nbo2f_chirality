"""Bayesian first-order FSS extrapolation of T_c(L) -> T_c(inf).

Model (linear in the parameters, so transforming the predictor 1/L^3 -- not the
response -- introduces no distortion):

    T_c(L) ~ Normal( T_inf + a / L^3 ,  se(L)^2 + sigma_sys^2 )

with flat priors on (T_inf, a) and sigma_sys ~ HalfNormal(scale = 1 K). The
extra per-size variance sigma_sys captures the convergence/estimator systematic
that the seed-to-seed SE underestimates (a fixed-sigma fit has reduced chi^2 ~
80). The coefficient integral is conjugate, so we marginalise sigma_sys on a
grid and report the posterior on T_inf as the mixture of the per-sigma
Gaussians.

T_c per size is read from the per-run estimator pipeline (tc_estimators.csv,
built by tc_estimators.py); nothing is hard-coded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EST = Path(__file__).resolve().parents[2] / "build" / "rewl" / "tc_estimators.csv"
SIGMA_SCALE = 1.0          # HalfNormal prior scale on sigma_sys (K), convergence-informed
SIG = np.linspace(1e-4, 6.0, 4000)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not EST.exists():
        raise SystemExit(f"missing {EST}; run  python {EST.parent / 'tc_estimators.py'}  first")
    g = pd.read_csv(EST).groupby("L")["tc_equal_area"]
    L = np.array(sorted(g.groups), float)
    return L, g.mean().loc[L].to_numpy(), g.std(ddof=1).loc[L].to_numpy(), g.count().loc[L].to_numpy()


def posterior(L: np.ndarray, T: np.ndarray, se: np.ndarray):
    """Marginalise sigma_sys; return (T_inf mean, T_inf sd, sigma_sys median, post over SIG)."""
    A = np.vstack([np.ones_like(L), 1.0 / L**3]).T
    t0 = np.empty_like(SIG); v0 = np.empty_like(SIG); logZ = np.empty_like(SIG)
    for i, s in enumerate(SIG):
        V = se**2 + s**2
        Lam = A.T @ (A / V[:, None])
        cov = np.linalg.inv(Lam)
        beta = cov @ (A.T @ (T / V))
        chi2 = np.sum((T - A @ beta) ** 2 / V)
        t0[i], v0[i] = beta[0], cov[0, 0]
        logZ[i] = -0.5 * np.sum(np.log(2 * np.pi * V)) - 0.5 * chi2 - 0.5 * np.linalg.slogdet(Lam)[1]
    lp = logZ - 0.5 * (SIG / SIGMA_SCALE) ** 2
    lp -= lp.max()
    post = np.exp(lp); post /= np.trapezoid(post, SIG)
    Et = np.trapezoid(post * t0, SIG)
    sdt = np.sqrt(np.trapezoid(post * (v0 + t0**2), SIG) - Et**2)
    cdf = np.cumsum(post); cdf /= cdf[-1]
    s_med = SIG[np.searchsorted(cdf, 0.5)]
    return Et, sdt, s_med, post, t0, v0, cdf


def main() -> None:
    L, T, std, n = load_data()
    se = std / np.sqrt(n)
    Et, sdt, s_med, post, t0, v0, cdf = posterior(L, T, se)

    # 95% credible interval on T_inf by sampling the mixture
    rng = np.random.default_rng(0)
    u = rng.random(200_000)
    sdraw = np.interp(u, cdf, SIG)
    tdraw = rng.normal(np.interp(sdraw, SIG, t0), np.sqrt(np.interp(sdraw, SIG, v0)))
    lo, hi = np.percentile(tdraw, [2.5, 97.5])
    s_lo, s_hi = SIG[np.searchsorted(cdf, [0.025, 0.975])]

    # MAP linear fit (sigma_sys at its posterior median) for the per-size residual table
    V = se**2 + s_med**2
    A = np.vstack([np.ones_like(L), 1.0 / L**3]).T
    beta = np.linalg.solve(A.T @ (A / V[:, None]), A.T @ (T / V))
    pred = A @ beta

    print("Bayesian FSS:  T_c(L) ~ N(T_inf + a/L^3, se^2 + sigma_sys^2)   (source: tc_estimators.csv)\n")
    print(f"{'L':>3} {'T_c (K)':>9} {'seedSE':>7} {'n':>2} {'fit (K)':>9} {'resid':>7}")
    for k, t, s, nn, p in zip(L, T, se, n, pred):
        print(f"{int(k):>3} {t:>9.2f} {s:>7.3f} {int(nn):>2} {p:>9.2f} {t - p:>+7.2f}")
    print()
    print(f"  T_c(inf) = {Et:.2f} +/- {sdt:.2f} K   (95% CI [{lo:.2f}, {hi:.2f}])   [posterior]")
    print(f"  sigma_sys = {s_med:.2f} K  (95% [{s_lo:.2f}, {s_hi:.2f}]);  prior HalfNormal({SIGMA_SCALE:.0f} K)")
    print()
    print("  leave-one-out T_c(inf) posterior mean:")
    for drop in L:
        keep = L != drop
        eti, _, _, *_ = posterior(L[keep], T[keep], se[keep])
        print(f"    without L={int(drop):>2}: {eti:.2f} K")


if __name__ == "__main__":
    main()
