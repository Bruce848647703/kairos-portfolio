"""风险度量模块：组合波动、Beta、VaR/CVaR 与风险贡献分解。

约定
----
- 收益序列输入为「每期收益」(Series/ndarray)，自动剔除 NaN；
- VaR / CVaR 一律返回**正数损失值**：0.02 表示该分位下的损失为 2%
  （极端情况下样本全为盈利时历史 VaR 可能为负，代表该分位无损失）；
- 正态分位数 norm_ppf 为自研实现（math.erf 二分法），参数 VaR 无需 scipy；
- weights 为 Series 且 cov 为 DataFrame 时自动按资产名对齐。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from ._util import Weights, align_pair, wrap_1d

SeriesLike = Union[pd.Series, np.ndarray]


def _as_series(returns: SeriesLike) -> np.ndarray:
    """收益序列归一：ravel + 剔除 NaN。"""
    arr = np.asarray(returns, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("收益序列为空")
    return arr


def _check_confidence(confidence: float) -> float:
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError(f"confidence 必须在 (0,1) 内，收到 {c}")
    return c


def norm_ppf(p: float) -> float:
    """标准正态分位函数（自研）：在 Φ(z)=0.5·erfc(−z/√2) 上做二分求根。

    100 次二分将 [−40, 40] 区间收敛到远超双精度的分辨率，确定且无依赖。
    """
    q = float(p)
    if not (0.0 < q < 1.0):
        raise ValueError(f"p 必须在 (0,1) 内，收到 {q}")

    def cdf(z: float) -> float:
        return 0.5 * math.erfc(-z / math.sqrt(2.0))

    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def portfolio_volatility(weights, cov, periods_per_year: float = 1.0) -> float:
    """组合波动率 σ_p = √(wᵀΣw)。

    periods_per_year 默认 1（返回每期波动）；日线协方差传 252 得年化波动。
    """
    w, sigma, _ = align_pair(weights, cov, "weights", "cov")
    var = max(float(w @ sigma @ w), 0.0)
    ppy = float(periods_per_year)
    if ppy <= 0:
        raise ValueError("periods_per_year 必须为正")
    return math.sqrt(var * ppy)


def beta(asset_returns: SeriesLike, market_returns: SeriesLike) -> float:
    """资产对市场的 Beta = Cov(r_a, r_m) / Var(r_m)（ddof 在比值中约掉）。"""
    a = _as_series(asset_returns)
    mk = _as_series(market_returns)
    if a.size != mk.size:
        raise ValueError(f"长度不一致：{a.size} vs {mk.size}")
    if a.size < 2:
        raise ValueError("至少需要 2 个观测")
    da = a - a.mean()
    dm = mk - mk.mean()
    var = float(dm @ dm)
    if var <= 0:
        raise ValueError("市场收益方差为零，Beta 无定义")
    return float(da @ dm) / var


def historical_var(returns: SeriesLike, confidence: float = 0.95) -> float:
    """历史模拟 VaR：损失 = −分位数 q_{1−c}(r)，返回正数损失值。

    分位数用线性插值（numpy 默认），因此对 confidence 单调不减。
    """
    r = _as_series(returns)
    c = _check_confidence(confidence)
    return -float(np.quantile(r, 1.0 - c))


def parametric_var(returns: SeriesLike = None, confidence: float = 0.95,
                   mean: float = None, std: float = None) -> float:
    """参数（正态假设）VaR：VaR = −(μ + z_{1−c}·σ)，返回正数损失值。

    可直接给 (mean, std)，或给收益序列自动估计（std 用 ddof=1）。
    正态分位数由自研 norm_ppf 计算，无需 scipy。
    """
    c = _check_confidence(confidence)
    if returns is not None:
        r = _as_series(returns)
        if r.size < 2:
            raise ValueError("至少需要 2 个观测")
        mean = float(r.mean())
        std = float(r.std(ddof=1))
    if mean is None or std is None:
        raise ValueError("必须提供 returns 或完整的 (mean, std)")
    if std < 0:
        raise ValueError("std 不能为负")
    z = norm_ppf(1.0 - c)
    return -(float(mean) + z * float(std))


def cvar(returns: SeriesLike, confidence: float = 0.95) -> float:
    """期望损失 ES/CVaR：损失超过 VaR 分位的尾部样本的平均损失（正数）。

    tail = {r ≤ q_{1−c}}，ES = −mean(tail) ≥ 同置信度 VaR（尾部均值不高于分位点）。
    """
    r = _as_series(returns)
    c = _check_confidence(confidence)
    threshold = -historical_var(r, c)      # 即 q_{1−c}
    tail = r[r <= threshold]
    if tail.size == 0:
        tail = r[r == r.min()]
    return float(-tail.mean())


@dataclass(frozen=True, eq=False)
class RiskContributions:
    """风险贡献分解结果（每项与输入 weights 同型：Series 或 ndarray）。

    字段
    ----
    marginal:    边际风险 ∂σ_p/∂w = Σw/σ_p；
    component:   成分风险 w ⊙ Σw/σ_p，其和恰等于 σ_p（欧拉分解）；
    pct:         成分风险占比 component/σ_p，其和为 1（风险平价的校验口径）；
    volatility:  组合每期波动 σ_p。
    """

    marginal: Weights
    component: Weights
    pct: Weights
    volatility: float


def risk_contributions(weights, cov) -> RiskContributions:
    """计算组合的边际/成分风险贡献，用于风险诊断与风险平价校验。"""
    w, sigma, index = align_pair(weights, cov, "weights", "cov")
    var = float(w @ sigma @ w)
    vol = math.sqrt(max(var, 0.0))
    if vol <= 1e-14:
        raise ValueError("组合波动接近零，风险贡献无定义")
    sw = sigma @ w
    marginal = sw / vol
    component = w * marginal
    pct = component / vol
    return RiskContributions(
        marginal=wrap_1d(marginal, index, name="marginal"),
        component=wrap_1d(component, index, name="component"),
        pct=wrap_1d(pct, index, name="pct"),
        volatility=vol,
    )
