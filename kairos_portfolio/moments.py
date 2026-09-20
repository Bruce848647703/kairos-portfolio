"""矩估计模块：均值收益与协方差矩阵估计。

数据约定：资产收益面板 DataFrame(index=日期, columns=资产)，也接受单资产 Series
或纯 ndarray。全部估计器为自研实现，仅依赖 numpy/pandas：

- mean_returns: 简单均值或指数加权均值 (EWM)；
- sample_cov:   样本协方差（ddof 可配）；
- ledoit_wolf:  Ledoit-Wolf 线性收缩协方差，支持「对角」与「常相关」两种目标，
                返回收缩后协方差与最优收缩强度 δ*；
- ema_cov:      指数加权协方差（RiskMetrics 风格时间衰减权重，近期样本更重要）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

ReturnsLike = Union[pd.DataFrame, pd.Series, np.ndarray]


def _clean_panel(returns: ReturnsLike) -> pd.DataFrame:
    """把输入归一成按日期排序、无缺失行的 float 面板。

    协方差/指数加权估计要求各行完整，直接整行剔除 NaN（而非成对剔除），
    保证结果确定且权重语义一致。
    """
    if isinstance(returns, pd.Series):
        df = returns.to_frame()
    elif isinstance(returns, pd.DataFrame):
        df = returns
    else:
        arr = np.asarray(returns, dtype="float64")
        if arr.ndim != 2:
            raise ValueError("returns 必须是二维面板 (index=日期, columns=资产)")
        df = pd.DataFrame(arr)
    df = df.astype("float64").dropna(how="any").sort_index()
    if df.empty or df.shape[1] == 0:
        raise ValueError("returns 面板在剔除 NaN 后为空")
    return df


def _ew_weights(n: int, halflife: Optional[float], span: Optional[float]) -> np.ndarray:
    """指数衰减权重：最近一期权重 1，往前第 k 期 λ^k，归一化后返回。

    参数换算与 pandas ewm 一致：halflife -> λ=0.5**(1/halflife)；span -> λ=1-2/(span+1)。
    halflife 与 span 必须恰好提供一个。
    """
    if (halflife is None) == (span is None):
        raise ValueError("halflife 与 span 必须恰好提供一个")
    if halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife 必须为正")
        lam = 0.5 ** (1.0 / float(halflife))
    else:
        if span <= 1:
            raise ValueError("span 必须大于 1")
        lam = 1.0 - 2.0 / (float(span) + 1.0)
    ages = np.arange(n - 1, -1, -1, dtype="float64")  # 各期距「现在」的期数
    w = lam ** ages
    return w / w.sum()


def mean_returns(returns: ReturnsLike, method: str = "simple",
                 halflife: Optional[float] = None,
                 span: Optional[float] = None) -> pd.Series:
    """资产均值收益估计。

    参数
    ----
    returns:  收益面板 (index=日期, columns=资产)。
    method:   "simple" 简单均值；"ewm" 指数加权均值（近期权重更高）。
    halflife / span: 仅 method="ewm" 时使用，二选一，含义同 pandas ewm。

    返回：以资产为索引的 Series。
    """
    ret = _clean_panel(returns)
    if method == "simple":
        return ret.mean()
    if method != "ewm":
        raise ValueError("method 只支持 'simple' 或 'ewm'")
    w = _ew_weights(len(ret), halflife, span)
    mu = w @ ret.to_numpy(dtype="float64")
    return pd.Series(mu, index=ret.columns)


def sample_cov(returns: ReturnsLike, ddof: int = 1) -> pd.DataFrame:
    """样本协方差矩阵：Σ = XᵀX/(n−ddof)，X 为去均值后的收益。

    ddof=1 为无偏估计（与 pandas .cov() 一致），ddof=0 为 MLE。
    """
    ret = _clean_panel(returns)
    n = len(ret)
    if n - ddof <= 0:
        raise ValueError(f"样本量 n={n} 不足以支撑 ddof={ddof}")
    x = ret.to_numpy(dtype="float64")
    x = x - x.mean(axis=0)
    cov = x.T @ x / float(n - ddof)
    cols = ret.columns
    return pd.DataFrame(cov, index=cols, columns=cols)


def _pi_hat(x: np.ndarray, s: np.ndarray, s_mle: np.ndarray) -> np.ndarray:
    """π̂_ij = (1/n)·Σ_t (x_ti·x_tj − s_ij)²，即 √n·s_ij 的渐近方差估计。

    x 必须已中心化；s_mle = XᵀX/n；s 可为任意 ddof 版本（展开式对两者都成立）。
    """
    n = x.shape[0]
    x2 = x * x
    m2 = x2.T @ x2 / n                       # (1/n)Σ_t x_ti²·x_tj²
    return m2 - 2.0 * s * s_mle + s * s


def _diag_target(s: np.ndarray) -> np.ndarray:
    """对角收缩目标：F = μI，μ = tr(S)/p 为平均方差。"""
    p = s.shape[0]
    return np.eye(p) * (float(np.trace(s)) / p)


def _const_corr_target(s: np.ndarray) -> np.ndarray:
    """常相关收缩目标：对角保留样本方差，非对角相关系数统一收缩到平均值 r̄。"""
    p = s.shape[0]
    d = np.diag(s).copy()
    u = np.sqrt(d)
    corr = s / np.outer(u, u)
    rbar = (float(corr.sum()) - p) / float(p * p - p)
    f = rbar * np.outer(u, u)
    np.fill_diagonal(f, d)
    return f


def _rho_const_corr(pi_hat: np.ndarray, s: np.ndarray) -> float:
    """常相关目标的 ρ̂ = Σ_ij Σ_kl (∂f_ij/∂s_kl)·π̂_kl，闭式向量化计算。

    推导要点（F 对 S 的依赖分两部分）：
    - 对角项 f_ii = s_ii            -> ρ̂_ii = π̂_ii；
    - 非对角项 f_ij = r̄·u_i·u_j（u_i=√s_ii）：
        ∂f_ij/∂s_kl 有两类贡献：
        (a) 经 r̄ 的间接导数，对全部 i≠j 相同，记 C = Σ_kl (∂r̄/∂s_kl)·π̂_kl；
        (b) 经 u_i、u_j 的直接导数，仅 k,l ∈ {i,j} 的对角元非零。
    求和后化简为仅依赖 u、r̄、π̂ 对角线的标量表达式。
    """
    p = s.shape[0]
    d = np.diag(s)
    u = np.sqrt(d)
    corr = s / np.outer(u, u)
    np.fill_diagonal(corr, 0.0)
    q = float(p * p - p)
    rbar = float(corr.sum()) / q

    # C = Σ_kl (∂r̄/∂s_kl)·π̂_kl：非对角 ∂r̄/∂s_kl = 1/(q·u_k·u_l)（有序对，π̂ 对称故
    # 求和自然覆盖两个方向）；对角 ∂r̄/∂s_kk = −(Σ_{l≠k} r_kl)/(q·s_kk)。
    off_pi = pi_hat / np.outer(u, u)
    np.fill_diagonal(off_pi, 0.0)
    diag_pi = np.diag(pi_hat)
    c_off = float(off_pi.sum()) / q
    c_diag = -float((diag_pi * corr.sum(axis=1) / d).sum()) / q
    c = c_off + c_diag

    # Σ_{i≠j} u_i·u_j = (Σu)² − tr(S)
    u_sum = float(u.sum())
    off_uu = u_sum * u_sum - float(d.sum())
    # Σ_{i≠j} [(r̄/2)(u_j/u_i)π̂_ii + (r̄/2)(u_i/u_j)π̂_jj] = r̄·Σ_i π̂_ii·(Σu/u_i − 1)
    mid = rbar * float((diag_pi * (u_sum / u - 1.0)).sum())
    rho_off = c * off_uu + mid
    return float(diag_pi.sum()) + rho_off


@dataclass(frozen=True, eq=False)
class LedoitWolfResult:
    """Ledoit-Wolf 收缩结果。

    字段
    ----
    cov:         收缩后协方差 Σ* = δ·F + (1−δ)·S；
    delta:       收缩强度 δ ∈ [0,1]，0 表示完全用样本协方差，1 表示完全用目标；
    target:      收缩目标矩阵 F；
    target_kind: "diag"（对角）或 "const_corr"（常相关）。
    """

    cov: pd.DataFrame
    delta: float
    target: pd.DataFrame
    target_kind: str


def ledoit_wolf(returns: ReturnsLike, target: str = "const_corr",
                ddof: int = 1, shrinkage: Optional[float] = None) -> LedoitWolfResult:
    """Ledoit-Wolf 线性收缩协方差估计（自研实现）。

    Σ* = δ·F + (1−δ)·S，S 为样本协方差，F 为结构化目标：
    - target="diag"：       F = (tr(S)/p)·I，把所有方差拉平、协方差清零；
    - target="const_corr"： 非对角相关系数统一收缩到平均值 r̄，对角保留样本方差。

    shrinkage=None 时按 Ledoit-Wolf 渐近 oracle 公式估计最优强度
    δ* = clip(κ̂/n, 0, 1)，κ̂ = (π̂ − ρ̂)/γ̂，其中 π̂ 为样本协方差各元素的
    渐近方差之和，ρ̂ 为目标对样本的导数修正项，γ̂ = ||F−S||²_F；
    传入 float 则直接指定 δ（便于做 δ=0 / δ=1 的端点对照实验）。

    单资产 (p=1) 时常相关目标退化，自动按对角目标处理（此时 δ*=0）。
    """
    if target not in ("diag", "const_corr"):
        raise ValueError("target 只支持 'diag' 或 'const_corr'")
    ret = _clean_panel(returns)
    x = ret.to_numpy(dtype="float64")
    n, p = x.shape
    if n < 2:
        raise ValueError("样本量至少为 2")
    if ddof < 0 or n - ddof <= 0:
        raise ValueError(f"ddof={ddof} 对样本量 n={n} 不合法")
    x = x - x.mean(axis=0)

    s_mle = x.T @ x / n                      # (1/n)Σ x xᵀ，π̂ 展开式需要
    s = s_mle * (n / float(n - ddof))        # 指定 ddof 的样本协方差
    if p == 1:
        target = "diag"
    if target == "diag":
        f = _diag_target(s)
    else:
        if np.any(np.diag(s) <= 0):
            raise ValueError("常相关目标要求各资产方差为正")
        f = _const_corr_target(s)

    if shrinkage is None:
        delta = _lw_optimal_delta(x, s, s_mle, f, target)
    else:
        delta = float(np.clip(float(shrinkage), 0.0, 1.0))

    cov = delta * f + (1.0 - delta) * s
    cols = ret.columns
    return LedoitWolfResult(
        cov=pd.DataFrame(cov, index=cols, columns=cols),
        delta=delta,
        target=pd.DataFrame(f, index=cols, columns=cols),
        target_kind=target,
    )


def _lw_optimal_delta(x: np.ndarray, s: np.ndarray, s_mle: np.ndarray,
                      f: np.ndarray, target: str) -> float:
    """按 LW oracle 公式计算最优收缩强度 δ* = clip(κ̂/n, 0, 1)。"""
    n = x.shape[0]
    pi = _pi_hat(x, s, s_mle)
    pi_sum = float(pi.sum())
    diff = f - s
    gamma = float((diff * diff).sum())       # γ̂ = ||F−S||²_F
    if gamma <= 1e-300:
        return 0.0                           # 样本与目标重合，无需收缩
    if target == "diag":
        # F = μI：∂f_ij/∂s_kl = δ_ij·δ_kl/p，逐项求和得 ρ̂ = Σ_i π̂_ii
        rho = float(np.trace(pi))
    else:
        rho = _rho_const_corr(pi, s)
    kappa = (pi_sum - rho) / gamma
    return float(min(max(kappa / n, 0.0), 1.0))


def ema_cov(returns: ReturnsLike, halflife: Optional[float] = None,
            span: Optional[float] = None, bias_correction: bool = True) -> pd.DataFrame:
    """指数加权协方差（RiskMetrics 风格）：越近的样本权重越高。

    权重 w_t ∝ λ^(距今期数)，归一化 Σw=1；先算加权均值再做加权二阶矩：
        Σ_ema = Σ_t w_t·(x_t−μ_w)(x_t−μ_w)ᵀ / (1 − Σ_t w_t²)
    分母 (1−Σw²) 为 Kish 有效样本修正（bias_correction=True 时启用）；
    当权重退化为均匀时结果精确等于 ddof=1 的样本协方差。

    halflife 与 span 必须恰好提供一个，换算同 pandas ewm。
    """
    ret = _clean_panel(returns)
    n = len(ret)
    if n < 2:
        raise ValueError("样本量至少为 2")
    w = _ew_weights(n, halflife, span)
    x = ret.to_numpy(dtype="float64")
    mu = w @ x
    xc = x - mu
    cov = (xc * w[:, None]).T @ xc
    if bias_correction:
        denom = 1.0 - float(w @ w)
        if denom <= 0:
            raise ValueError("权重过度集中（有效样本 < 2），无法做偏差修正")
        cov = cov / denom
    cols = ret.columns
    return pd.DataFrame(cov, index=cols, columns=cols)
