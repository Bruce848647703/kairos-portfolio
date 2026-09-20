"""测试共用夹具：固定 seed 的合成收益面板与派生协方差/期望收益（离线可复现）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ASSETS = ["AAA", "BBB", "CCC"]


@pytest.fixture()
def returns_panel() -> pd.DataFrame:
    """3 资产 400 日合成收益：2 个公共因子 + 不同波动的特质噪声。"""
    rng = np.random.default_rng(20260920)
    n = 400
    f = rng.standard_normal((n, 2))
    load = np.array([[1.0, 0.2], [0.7, 0.5], [0.3, 0.9]])
    idio = rng.standard_normal((n, 3)) * np.array([0.8, 1.0, 1.4])
    rets = (f @ load.T + idio) * 0.01
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rets, index=idx, columns=ASSETS)


@pytest.fixture()
def cov3(returns_panel) -> pd.DataFrame:
    """由合成面板得到的 3 资产样本协方差。"""
    from kairos_portfolio import sample_cov

    return sample_cov(returns_panel)


@pytest.fixture()
def mu3() -> pd.Series:
    """固定期望收益（日频、明显递减），避免随机噪声干扰优化断言。"""
    return pd.Series([0.0008, 0.0005, 0.0002], index=ASSETS)


@pytest.fixture()
def cov4() -> pd.DataFrame:
    """4 资产协方差（低秩因子 + 对角），用于风险预算等测试。"""
    cols = ["W", "X", "Y", "Z"]
    rng = np.random.default_rng(11)
    a = rng.standard_normal((4, 3))
    sigma = (a @ a.T) / 3.0 + np.diag([0.5, 1.0, 1.5, 2.0])
    sigma *= 1e-4
    return pd.DataFrame(sigma, index=cols, columns=cols)
