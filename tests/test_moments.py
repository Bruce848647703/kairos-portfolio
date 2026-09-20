"""moments 模块测试：均值/样本协方差/Ledoit-Wolf 收缩/指数加权协方差。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_portfolio import ema_cov, ledoit_wolf, mean_returns, sample_cov


# ---- 均值 ----

def test_mean_returns_simple_matches_manual(returns_panel):
    mu = mean_returns(returns_panel)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(returns_panel.columns)
    np.testing.assert_allclose(mu.to_numpy(), returns_panel.to_numpy().mean(axis=0),
                               rtol=0, atol=1e-15)


def test_mean_returns_ewm_matches_pandas(returns_panel):
    mu = mean_returns(returns_panel, method="ewm", halflife=20)
    ref = returns_panel.ewm(halflife=20, adjust=True).mean().iloc[-1]
    np.testing.assert_allclose(mu.to_numpy(), ref.to_numpy(), rtol=1e-8)


def test_mean_returns_ewm_follows_recent_regime():
    n = 250
    vals = np.concatenate([np.zeros(200), np.full(50, 0.01)])
    df = pd.DataFrame({"A": vals}, index=pd.bdate_range("2021-01-01", periods=n))
    simple = float(mean_returns(df).iloc[0])
    ewm = float(mean_returns(df, method="ewm", halflife=10).iloc[0])
    assert simple == pytest.approx(0.002)
    assert ewm > 2 * simple  # 指数加权应显著跟随近期均值抬升


def test_mean_returns_validates_method(returns_panel):
    with pytest.raises(ValueError):
        mean_returns(returns_panel, method="magic")
    with pytest.raises(ValueError):
        mean_returns(returns_panel, method="ewm")                    # 缺 halflife/span
    with pytest.raises(ValueError):
        mean_returns(returns_panel, method="ewm", halflife=5, span=5)  # 二选一


# ---- 样本协方差 ----

def test_sample_cov_matches_pandas(returns_panel):
    cov = sample_cov(returns_panel)
    assert list(cov.columns) == list(returns_panel.columns)
    np.testing.assert_allclose(cov.to_numpy(), returns_panel.cov().to_numpy(), rtol=1e-10)


def test_sample_cov_symmetric_psd_and_ddof_scaling(returns_panel):
    n = len(returns_panel)
    c1 = sample_cov(returns_panel, ddof=1).to_numpy()
    c0 = sample_cov(returns_panel, ddof=0).to_numpy()
    assert np.allclose(c1, c1.T)
    assert np.linalg.eigvalsh(c1).min() > -1e-18
    np.testing.assert_allclose(c0, c1 * ((n - 1) / n), rtol=1e-12)


# ---- Ledoit-Wolf ----

def test_ledoit_wolf_delta_in_range_and_psd(returns_panel):
    for tgt in ("diag", "const_corr"):
        res = ledoit_wolf(returns_panel, target=tgt)
        assert 0.0 <= res.delta <= 1.0
        cov = res.cov.to_numpy()
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() >= -1e-18
        assert res.target_kind == tgt


def test_ledoit_wolf_delta_zero_is_sample_cov(returns_panel):
    for tgt in ("diag", "const_corr"):
        res = ledoit_wolf(returns_panel, target=tgt, shrinkage=0.0)
        assert res.delta == 0.0
        np.testing.assert_allclose(res.cov.to_numpy(),
                                   sample_cov(returns_panel).to_numpy(),
                                   rtol=1e-12, atol=1e-20)


def test_ledoit_wolf_delta_one_diag_target(returns_panel):
    res = ledoit_wolf(returns_panel, target="diag", shrinkage=1.0)
    cov = res.cov.to_numpy()
    s = sample_cov(returns_panel).to_numpy()
    off = cov - np.diag(np.diag(cov))
    assert np.allclose(off, 0.0, atol=1e-20)                      # 非对角清零
    assert np.allclose(np.diag(cov), np.trace(s) / s.shape[0], rtol=1e-12)  # 对角=平均方差


def test_ledoit_wolf_delta_one_const_corr_target(returns_panel):
    res = ledoit_wolf(returns_panel, target="const_corr", shrinkage=1.0)
    cov = res.cov.to_numpy()
    s = sample_cov(returns_panel).to_numpy()
    np.testing.assert_allclose(np.diag(cov), np.diag(s), rtol=1e-12)   # 对角保留样本方差
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert np.allclose(off, off[0], rtol=1e-8)                          # 非对角相关恒定


def test_ledoit_wolf_shrinkage_improves_accuracy():
    """因子结构 + 小样本多资产：多种子平均下 LW 的均方误差应小于样本协方差。

    单次实现存在抽样波动，LW 的保证是期望意义上的 MSE 改善，故对 12 个
    固定 seed 的独立实验取平均（离线可复现）。
    """
    n, p = 60, 10
    err = {"sample": [], "diag": [], "const_corr": []}
    for sd in range(12):
        rng = np.random.default_rng(100 + sd)
        load = rng.standard_normal((p, 2)) * 0.7
        x = rng.standard_normal((n, 2)) @ load.T + rng.standard_normal((n, p)) * 0.6
        sigma_true = load @ load.T + 0.36 * np.eye(p)   # 生成过程的解析真值
        df = pd.DataFrame(x)
        err["sample"].append(
            np.linalg.norm(sample_cov(df).to_numpy() - sigma_true, "fro") ** 2)
        for tgt in ("diag", "const_corr"):
            lw = ledoit_wolf(df, target=tgt)
            err[tgt].append(np.linalg.norm(lw.cov.to_numpy() - sigma_true, "fro") ** 2)
    mean_err = {k: float(np.mean(v)) for k, v in err.items()}
    assert mean_err["diag"] < mean_err["sample"]
    assert mean_err["const_corr"] < mean_err["sample"]


def test_ledoit_wolf_diag_target_on_independent_assets():
    """独立资产（真协方差为对角）：δ* 应显著为正，且收缩大幅降低估计误差。"""
    n, p = 50, 8
    err_s, err_lw, deltas = [], [], []
    for sd in range(5):
        rng = np.random.default_rng(200 + sd)
        d = rng.uniform(0.5, 2.0, p)
        x = rng.standard_normal((n, p)) * np.sqrt(d)
        df = pd.DataFrame(x)
        res = ledoit_wolf(df, target="diag")
        deltas.append(res.delta)
        err_s.append(np.linalg.norm(sample_cov(df).to_numpy() - np.diag(d), "fro") ** 2)
        err_lw.append(np.linalg.norm(res.cov.to_numpy() - np.diag(d), "fro") ** 2)
    assert min(deltas) > 0.05                          # 应识别出明显的收缩需求
    assert float(np.mean(err_lw)) < 0.5 * float(np.mean(err_s))   # 误差大幅下降


def test_rho_diag_matches_numeric_derivative(returns_panel):
    """白盒校验：对角目标 ρ̂ = tr(π̂) 应等于 Σf(S+tπ̂) 对 t 的方向导数。"""
    from kairos_portfolio.moments import _diag_target, _pi_hat

    x = returns_panel.to_numpy() * 100.0              # 放大量纲，保证数值求导条件数
    x = x - x.mean(axis=0)
    n = x.shape[0]
    s = x.T @ x / n
    pi = _pi_hat(x, s, s)
    t = 1e-5
    num = (_diag_target(s + t * pi).sum() - _diag_target(s - t * pi).sum()) / (2 * t)
    assert float(np.trace(pi)) == pytest.approx(num, rel=1e-6)


def test_rho_const_corr_matches_numeric_derivative(returns_panel):
    """白盒校验：常相关目标 ρ̂ 闭式解应等于 Σf(S+tπ̂) 对 t 的方向导数。"""
    from kairos_portfolio.moments import _const_corr_target, _pi_hat, _rho_const_corr

    x = returns_panel.to_numpy() * 100.0
    x = x - x.mean(axis=0)
    n = x.shape[0]
    s = x.T @ x / n
    pi = _pi_hat(x, s, s)
    t = 1e-5
    num = (_const_corr_target(s + t * pi).sum() - _const_corr_target(s - t * pi).sum()) / (2 * t)
    rho = _rho_const_corr(pi, s)
    assert rho == pytest.approx(num, rel=1e-5)


def test_ledoit_wolf_validates_target(returns_panel):
    with pytest.raises(ValueError):
        ledoit_wolf(returns_panel, target="unknown")


# ---- 指数加权协方差 ----

def test_ema_cov_long_halflife_approaches_sample_cov(returns_panel):
    ema = ema_cov(returns_panel, halflife=1e6).to_numpy()
    s = sample_cov(returns_panel).to_numpy()
    np.testing.assert_allclose(ema, s, rtol=1e-3)


def test_ema_cov_matches_manual_two_asset():
    rets = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.00], "B": [0.02, 0.01, -0.01, 0.04]})
    cov = ema_cov(rets, span=3)
    lam = 1.0 - 2.0 / (3.0 + 1.0)                    # span=3 -> λ=0.5
    w = np.array([lam ** 3, lam ** 2, lam, 1.0])
    w = w / w.sum()
    x = rets.to_numpy()
    xc = x - w @ x
    manual = (xc * w[:, None]).T @ xc / (1.0 - float(w @ w))
    np.testing.assert_allclose(cov.to_numpy(), manual, rtol=1e-10, atol=1e-18)


def test_ema_cov_symmetric_psd(returns_panel):
    cov = ema_cov(returns_panel, halflife=10).to_numpy()
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-20


def test_ema_cov_param_validation(returns_panel):
    with pytest.raises(ValueError):
        ema_cov(returns_panel)                        # 缺 halflife/span
    with pytest.raises(ValueError):
        ema_cov(returns_panel, halflife=5, span=5)    # 二选一
