"""optimize 模块测试：可行性(和为1/bounds)、最优性(对比等权与解析解)、风险平价与目标波动。"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

from kairos_portfolio import (
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    portfolio_volatility,
    risk_contributions,
    risk_parity,
    target_volatility,
)


# ---- 等权 ----

def test_equal_weight_variants():
    w = equal_weight(4)
    assert isinstance(w, np.ndarray)
    assert float(w.sum()) == pytest.approx(1.0)
    assert np.all(w >= 0)

    s = equal_weight(pd.Index(["A", "B"]))
    assert isinstance(s, pd.Series)
    assert list(s.index) == ["A", "B"]
    assert float(s.sum()) == pytest.approx(1.0)

    df = pd.DataFrame(np.eye(3), columns=["X", "Y", "Z"])
    s2 = equal_weight(df)
    assert list(s2.index) == ["X", "Y", "Z"]


# ---- 最小方差 ----

def test_min_variance_beats_equal_weight(cov3):
    ew = equal_weight(3)
    v_ew = portfolio_volatility(ew, cov3)

    mv = np.asarray(min_variance(cov3, long_only=False))
    assert float(mv.sum()) == pytest.approx(1.0)
    v_mv = portfolio_volatility(mv, cov3)
    assert v_mv <= v_ew + 1e-12                       # 无约束最小方差 <= 等权

    mv_lo = np.asarray(min_variance(cov3, long_only=True))
    assert float(mv_lo.sum()) == pytest.approx(1.0, abs=1e-6)
    assert np.all(mv_lo >= -1e-9)                     # long-only 可行
    v_lo = portfolio_volatility(mv_lo, cov3)
    assert v_lo <= v_ew + 1e-9
    assert v_lo >= v_mv - 1e-9                        # 受约束不会优于无约束


def test_min_variance_analytic_matches_inverse(cov3):
    sigma = cov3.to_numpy()
    mv = np.asarray(min_variance(cov3, long_only=False))
    ref = np.linalg.solve(sigma, np.ones(3))
    ref = ref / ref.sum()
    np.testing.assert_allclose(mv, ref, rtol=1e-8)


def test_min_variance_respects_bounds(cov3):
    bounds = [(0.0, 0.2), (0.0, 1.0), (0.4, 0.8)]
    w = np.asarray(min_variance(cov3, long_only=True, bounds=bounds))
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    for wi, (lo, hi) in zip(w, bounds):
        assert lo - 1e-6 <= wi <= hi + 1e-6


# ---- 最大夏普 ----

def test_max_sharpe_unconstrained_matches_tangent_direction(cov3, mu3):
    w = np.asarray(max_sharpe(mu3, cov3, risk_free=0.0001, long_only=False))
    sigma = cov3.to_numpy()
    z = np.linalg.solve(sigma, mu3.to_numpy() - 0.0001)
    ref = z / z.sum()                                 # 解析切点 w ∝ Σ⁻¹(μ−rf)
    np.testing.assert_allclose(w, ref, rtol=1e-8)
    assert float(w.sum()) == pytest.approx(1.0)


def test_max_sharpe_long_only_feasible_and_not_better_than_free(cov3, mu3):
    w_free = np.asarray(max_sharpe(mu3, cov3, risk_free=0.0, long_only=False))
    w_lo = np.asarray(max_sharpe(mu3, cov3, risk_free=0.0, long_only=True))
    assert float(w_lo.sum()) == pytest.approx(1.0, abs=1e-6)
    assert np.all(w_lo >= -1e-9)

    sigma = cov3.to_numpy()
    m = mu3.to_numpy()

    def sharpe(w):
        return float(m @ w) / math.sqrt(float(w @ sigma @ w))

    assert sharpe(w_lo) <= sharpe(w_free) + 1e-6      # 可行域更小，夏普不会更高


# ---- 均值-方差 ----

def test_mean_variance_lambda_monotonic_risk(cov3, mu3):
    vols = []
    for lam in (2.0, 10.0, 50.0):
        w = np.asarray(mean_variance(mu3, cov3, risk_aversion=lam))   # 默认 long-only
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-9)
        vols.append(portfolio_volatility(w, cov3))
    assert vols[0] >= vols[1] - 1e-9 >= vols[2] - 2e-9  # λ 越大越厌恶风险 -> 波动越低


def test_mean_variance_target_return_hits_target(cov3, mu3):
    target = 0.5 * (float(mu3.min()) + float(mu3.max()))
    w = np.asarray(mean_variance(mu3, cov3, target_return=target, long_only=False))
    assert float(w @ mu3.to_numpy()) == pytest.approx(target, abs=1e-7)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-7)


def test_mean_variance_analytic_matches_slqp(cov3, mu3):
    wa = np.asarray(mean_variance(mu3, cov3, risk_aversion=8.0, long_only=False))
    # 给一个宽松 bounds 强制走 SLSQP 路径，应与闭式解一致
    wb = np.asarray(mean_variance(mu3, cov3, risk_aversion=8.0,
                                  long_only=False, bounds=(-5.0, 5.0)))
    np.testing.assert_allclose(wb, wa, atol=1e-4)


def test_mean_variance_infeasible_target_raises(cov3, mu3):
    with pytest.raises(RuntimeError):
        mean_variance(mu3, cov3, target_return=0.05, long_only=True)  # 远超最大期望收益


def test_mean_variance_requires_exactly_one_objective(cov3, mu3):
    with pytest.raises(ValueError):
        mean_variance(mu3, cov3)
    with pytest.raises(ValueError):
        mean_variance(mu3, cov3, risk_aversion=2.0, target_return=0.0005)


# ---- 风险平价 ----

def test_risk_parity_equal_contributions(cov3):
    w = np.asarray(risk_parity(cov3))
    assert float(w.sum()) == pytest.approx(1.0)
    assert np.all(w > 0)                              # 天然 long-only
    pct = np.asarray(risk_contributions(w, cov3).pct)
    np.testing.assert_allclose(pct, np.full(3, 1.0 / 3.0), rtol=1e-3)


def test_risk_parity_custom_budget(cov4):
    b = np.array([0.4, 0.3, 0.2, 0.1])
    w = np.asarray(risk_parity(cov4, risk_budget=b))
    assert float(w.sum()) == pytest.approx(1.0)
    pct = np.asarray(risk_contributions(w, cov4).pct)
    np.testing.assert_allclose(pct, b, rtol=1e-3)


def test_risk_parity_identity_cov_is_equal_weight():
    cols = list("ABC")
    cov = pd.DataFrame(np.eye(3) * 0.01, index=cols, columns=cols)
    w = np.asarray(risk_parity(cov))
    np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-6)


def test_risk_parity_validates_budget(cov3):
    with pytest.raises(ValueError):
        risk_parity(cov3, risk_budget=np.array([0.5, -0.2, 0.7]))
    with pytest.raises(ValueError):
        risk_parity(cov3, risk_budget=np.array([0.5, 0.5]))   # 维度不符


# ---- 目标波动 ----

def test_target_volatility_scales_and_caps(cov3):
    base = equal_weight(3)
    vol0 = portfolio_volatility(base, cov3)

    tv = target_volatility(base, cov3, target_vol=vol0 / 2.0)
    assert tv.exposure == pytest.approx(0.5, rel=1e-6)
    assert tv.cash_weight == pytest.approx(0.5, rel=1e-6)
    assert tv.portfolio_vol == pytest.approx(vol0 / 2.0, rel=1e-6)
    assert float(np.asarray(tv.weights).sum()) == pytest.approx(0.5, rel=1e-6)

    capped = target_volatility(base, cov3, target_vol=vol0 * 100.0, max_leverage=1.0)
    assert capped.exposure == pytest.approx(1.0)
    assert capped.cash_weight == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.asarray(capped.weights), np.asarray(base), atol=1e-12)


def test_target_volatility_allows_leverage(cov3):
    base = equal_weight(3)
    vol0 = portfolio_volatility(base, cov3)
    tv = target_volatility(base, cov3, target_vol=vol0 * 1.5, max_leverage=2.0)
    assert tv.exposure == pytest.approx(1.5, rel=1e-6)
    assert tv.portfolio_vol == pytest.approx(vol0 * 1.5, rel=1e-6)
    assert tv.cash_weight == pytest.approx(-0.5, rel=1e-6)   # 加杠杆 -> 负现金


# ---- 类型约定与 scipy 降级 ----

def test_ndarray_in_ndarray_out():
    sigma = np.eye(2) * 0.01
    w = min_variance(sigma, long_only=False)
    assert isinstance(w, np.ndarray)
    mu = np.array([0.001, 0.002])
    w2 = max_sharpe(mu, sigma, long_only=False)
    assert isinstance(w2, np.ndarray)


def test_series_index_preserved(cov3, mu3):
    w = min_variance(cov3, long_only=False)
    assert isinstance(w, pd.Series)
    assert list(w.index) == list(cov3.columns)


def test_missing_scipy_raises_clear_error(monkeypatch, cov3, mu3):
    """scipy 缺失：约束路径给出清晰 ImportError，解析路径不受影响。"""
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(ImportError, match="scipy"):
        min_variance(cov3, long_only=True)
    with pytest.raises(ImportError, match="scipy"):
        mean_variance(mu3, cov3, target_return=0.0005, long_only=True)
    w = min_variance(cov3, long_only=False)           # 解析解无需 scipy
    assert float(np.sum(w)) == pytest.approx(1.0)
    w2 = mean_variance(mu3, cov3, risk_aversion=5.0, long_only=False)
    assert float(np.sum(w2)) == pytest.approx(1.0, abs=1e-6)
