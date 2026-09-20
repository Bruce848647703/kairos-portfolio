"""risk 模块测试：波动、Beta、VaR/CVaR 单调性与大小关系、风险贡献分解。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kairos_portfolio import (
    beta,
    cvar,
    historical_var,
    norm_ppf,
    parametric_var,
    portfolio_volatility,
    risk_contributions,
)


# ---- 组合波动 ----

def test_portfolio_volatility_single_asset():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    assert portfolio_volatility([1.0], cov) == pytest.approx(0.2)


def test_portfolio_volatility_matches_quadratic_form(cov3):
    w = np.array([0.2, 0.3, 0.5])
    sigma = cov3.to_numpy()
    expect = math.sqrt(float(w @ sigma @ w))
    assert portfolio_volatility(w, cov3) == pytest.approx(expect, rel=1e-12)
    # 年化
    assert portfolio_volatility(w, cov3, periods_per_year=252) == pytest.approx(expect * math.sqrt(252))


# ---- Beta ----

def test_beta_known_values():
    rng = np.random.default_rng(5)
    mkt = pd.Series(rng.standard_normal(2000) * 0.01)
    tiny = pd.Series(rng.standard_normal(2000) * 1e-7)
    assert beta(mkt, mkt) == pytest.approx(1.0)
    assert beta(2.0 * mkt + tiny, mkt) == pytest.approx(2.0, abs=1e-3)
    assert beta(-1.5 * mkt + tiny, mkt) == pytest.approx(-1.5, abs=1e-3)


def test_beta_validates_input():
    with pytest.raises(ValueError):
        beta(pd.Series([0.01, 0.02]), pd.Series([0.01]))       # 长度不一致
    with pytest.raises(ValueError):
        beta(pd.Series([0.01, 0.02]), pd.Series([0.0, 0.0]))   # 市场零方差


# ---- VaR / CVaR ----

def test_historical_var_monotonic_in_confidence():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_normal(5000) * 0.01 - 0.0002)
    v90 = historical_var(r, 0.90)
    v95 = historical_var(r, 0.95)
    v99 = historical_var(r, 0.99)
    assert v90 < v95 < v99                      # 置信度越高，损失分位越极端
    assert v95 > 0


def test_cvar_at_least_var_and_monotonic():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.standard_t(4, 5000) * 0.01)   # 厚尾，尾部差异更明显
    for c in (0.90, 0.95, 0.99):
        es = cvar(r, c)
        v = historical_var(r, c)
        assert es >= v - 1e-12                      # ES 不劣于同分位 VaR
    assert cvar(r, 0.90) <= cvar(r, 0.95) <= cvar(r, 0.99)


def test_historical_var_cvar_small_case_exact():
    r = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04])
    # 1−c=0.1 分位（线性插值）= -0.041 -> VaR = 0.041
    assert historical_var(r, 0.90) == pytest.approx(0.041, rel=1e-9)
    # 尾部 = {-0.05, -0.041?}: r <= -0.041 只有 -0.05 -> ES = 0.05
    assert cvar(r, 0.90) == pytest.approx(0.05, rel=1e-9)
    assert cvar(r, 0.90) >= historical_var(r, 0.90)


def test_parametric_var_formula_and_ppf():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-10)
    assert norm_ppf(0.975) == pytest.approx(1.959963985, rel=1e-8)
    assert norm_ppf(0.95) == pytest.approx(-norm_ppf(0.05), rel=1e-10)

    r = np.array([0.01, -0.02, 0.005, -0.003, 0.002])
    z = norm_ppf(0.05)                              # ≈ -1.6449
    expect = -(r.mean() + z * r.std(ddof=1))
    assert parametric_var(r, 0.95) == pytest.approx(expect, rel=1e-10)
    # 直接给 (mean, std)
    assert parametric_var(mean=0.0, std=1.0, confidence=0.95) == pytest.approx(1.6448536, rel=1e-6)


def test_parametric_var_close_to_historical_on_normal():
    rng = np.random.default_rng(15)
    r = pd.Series(rng.standard_normal(20000) * 0.01)
    v_h = historical_var(r, 0.95)
    v_p = parametric_var(r, 0.95)
    assert v_p == pytest.approx(v_h, rel=0.05)      # 正态样本下两者应接近


def test_confidence_validation():
    r = pd.Series([0.01, -0.01])
    with pytest.raises(ValueError):
        historical_var(r, 1.5)
    with pytest.raises(ValueError):
        cvar(r, 0.0)


# ---- 风险贡献 ----

def test_risk_contributions_decompose_vol(cov3):
    w = np.array([0.2, 0.3, 0.5])
    rc = risk_contributions(w, cov3)
    sigma = cov3.to_numpy()
    vol = math.sqrt(float(w @ sigma @ w))
    assert rc.volatility == pytest.approx(vol, rel=1e-12)
    assert float(np.sum(rc.component)) == pytest.approx(vol, rel=1e-10)   # 欧拉分解
    assert float(np.sum(rc.pct)) == pytest.approx(1.0, rel=1e-10)
    np.testing.assert_allclose(np.asarray(rc.marginal), sigma @ w / vol, rtol=1e-10)


def test_risk_contributions_series_index(cov3):
    w = pd.Series([1.0 / 3.0] * 3, index=cov3.columns)
    rc = risk_contributions(w, cov3)
    assert isinstance(rc.pct, pd.Series)
    assert list(rc.pct.index) == list(cov3.columns)


def test_risk_contributions_misaligned_series_is_reordered(cov3):
    w = pd.Series([0.5, 0.3, 0.2], index=list(reversed(cov3.columns)))
    rc = risk_contributions(w, cov3)
    # 按 cov 列名对齐后，pct 索引应与 cov3 一致且占比和为 1
    assert list(rc.pct.index) == list(cov3.columns)
    assert float(np.sum(rc.pct)) == pytest.approx(1.0, rel=1e-10)
    w_aligned = w.reindex(cov3.columns)
    np.testing.assert_allclose(np.asarray(rc.marginal),
                               cov3.to_numpy() @ w_aligned.to_numpy() / rc.volatility,
                               rtol=1e-10)


def test_risk_contributions_zero_vol_raises(cov3):
    with pytest.raises(ValueError):
        risk_contributions(np.zeros(3), cov3)
