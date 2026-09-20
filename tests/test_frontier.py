"""frontier 模块测试：有效前沿的可行性、单调性与输出形态。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_portfolio import efficient_frontier, min_variance, portfolio_volatility


def test_frontier_monotonic_vol(cov3, mu3):
    fr = efficient_frontier(mu3, cov3, n_points=8, long_only=True)
    df = fr.frame()
    assert len(df) >= 5
    rets = df["target_return"].to_numpy()
    vols = df["volatility"].to_numpy()
    assert np.all(np.diff(rets) >= 0)                 # 目标收益升序
    assert np.all(np.diff(vols) >= -1e-9)             # 有效段：更高收益对应更高波动
    assert vols[-1] > vols[0] + 1e-9                  # 且非常数


def test_frontier_weights_feasible(cov3, mu3):
    fr = efficient_frontier(mu3, cov3, n_points=5, long_only=True)
    wf = fr.weights_frame()
    assert isinstance(wf, pd.DataFrame)
    assert wf.shape == (len(fr), 3)
    assert np.allclose(wf.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
    assert (wf.to_numpy() >= -1e-7).all()             # long-only 可行


def test_frontier_vol_not_below_min_variance(cov3, mu3):
    w_mv = np.asarray(min_variance(cov3, long_only=True))
    v_mv = portfolio_volatility(w_mv, cov3)
    fr = efficient_frontier(mu3, cov3, n_points=6, long_only=True)
    assert np.all(fr.frame()["volatility"].to_numpy() >= v_mv - 1e-7)


def test_frontier_explicit_targets_unconstrained(cov3, mu3):
    # 显式目标收益取在有效段内（最小方差组合收益之上），乱序传入应自动排序
    w_mv = np.asarray(min_variance(cov3, long_only=False))
    rt_mv = float(w_mv @ mu3.to_numpy())
    gap = float(mu3.max()) - rt_mv
    targets = [rt_mv + 0.75 * gap, rt_mv + 0.25 * gap, rt_mv + 0.5 * gap]
    fr = efficient_frontier(mu3, cov3, target_returns=targets, long_only=False)
    df = fr.frame()
    assert len(df) == 3
    np.testing.assert_allclose(df["target_return"].to_numpy(), sorted(targets))
    vols = df["volatility"].to_numpy()
    assert np.all(np.diff(vols) >= -1e-9)             # 有效段波动随目标收益不减


def test_frontier_infeasible_targets_are_skipped(cov3, mu3):
    # 0.05 远超 long-only 可达收益，应被跳过而非中断
    fr = efficient_frontier(mu3, cov3, target_returns=[0.0004, 0.05], long_only=True)
    assert len(fr.points) == 1
    assert fr.failed_targets == (0.05,)


def test_frontier_rejects_degenerate_mu(cov3):
    mu_flat = pd.Series([0.001] * 3, index=cov3.columns)
    with pytest.raises(ValueError):
        efficient_frontier(mu_flat, cov3, n_points=4, long_only=True)


def test_frontier_ndarray_io():
    rng = np.random.default_rng(21)
    a = rng.standard_normal((3, 2))
    sigma = a @ a.T + np.eye(3)
    mu = np.array([0.002, 0.001, 0.0005])
    fr = efficient_frontier(mu, sigma, n_points=4, long_only=False)
    assert isinstance(fr.points[0].weights, np.ndarray)
    assert len(fr.frame()) == len(fr)
