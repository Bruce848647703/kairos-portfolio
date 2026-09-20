"""有效前沿模块。

在一组目标收益上逐点求解均值-方差优化（min wᵀΣw s.t. μᵀw=r, Σw=1），
得到 (目标收益, 波动, 权重) 序列。默认目标收益区间取
[同约束下最小方差组合的收益, 资产期望收益最大值]，恰好覆盖「更高收益必然
伴随更高波动」的单调有效段；区间外的目标收益（无效段）求解失败会被跳过并
记录在 failed_targets 中。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._util import Weights, align_pair
from .optimize import BoundsLike, mean_variance, min_variance


@dataclass(frozen=True, eq=False)
class FrontierPoint:
    """有效前沿上的一个点：目标收益、对应最小波动与最优权重。"""

    target_return: float
    volatility: float
    weights: Weights


@dataclass(eq=False)
class Frontier:
    """有效前沿点集。points 按目标收益升序；failed_targets 记录不可行目标。"""

    points: List[FrontierPoint]
    failed_targets: Tuple[float, ...] = ()

    def __len__(self) -> int:
        return len(self.points)

    def frame(self) -> pd.DataFrame:
        """两列 DataFrame：target_return 与 volatility，便于打印/画图。"""
        return pd.DataFrame({
            "target_return": [p.target_return for p in self.points],
            "volatility": [p.volatility for p in self.points],
        })

    def weights_frame(self) -> pd.DataFrame:
        """权重矩阵：index=目标收益，columns=资产（ndarray 权重时为 0..n−1）。"""
        if not self.points:
            raise ValueError("前沿没有有效点")
        rows = []
        for p in self.points:
            w = p.weights
            rows.append(w if isinstance(w, pd.Series) else pd.Series(np.asarray(w, dtype="float64")))
        out = pd.DataFrame(rows, index=[p.target_return for p in self.points])
        out.index.name = "target_return"
        return out


def efficient_frontier(mu, cov, target_returns: Optional[np.ndarray] = None,
                       n_points: int = 10, long_only: bool = True,
                       bounds: BoundsLike = None) -> Frontier:
    """构造有效前沿。

    参数
    ----
    mu / cov:       期望收益与协方差（ndarray 或带资产索引的 pandas 对象）。
    target_returns: 显式目标收益序列（自动排序去重）；None 时取默认区间
                    [最小方差组合收益, max(mu)] 上的 n_points 个等距点。
    n_points:       默认区间的采样点数（≥2）。
    long_only / bounds: 传给每个 mean_variance 子问题的约束设定。

    返回：Frontier；个别目标不可行（超出收益可达范围等）时跳过并计入 failed_targets，
    全部失败才抛 RuntimeError。long_only=True 时需要 scipy。
    """
    m, sigma, _ = align_pair(mu, cov, "mu", "cov")
    if target_returns is None:
        if int(n_points) < 2:
            raise ValueError("n_points 至少为 2")
        w_mv = np.asarray(min_variance(cov, long_only=long_only, bounds=bounds), dtype="float64")
        rt_lo = float(m @ w_mv)
        rt_hi = float(m.max())
        if rt_hi <= rt_lo + 1e-12:
            raise ValueError("最小方差组合收益已不低于最大期望收益，无法构造上升的有效前沿")
        targets = np.linspace(rt_lo, rt_hi, int(n_points))
    else:
        targets = np.unique(np.asarray(target_returns, dtype="float64").ravel())
        if targets.size == 0 or not np.all(np.isfinite(targets)):
            raise ValueError("target_returns 为空或含 NaN/Inf")

    points: List[FrontierPoint] = []
    failed: List[float] = []
    for t in targets:
        t = float(t)
        try:
            w = mean_variance(mu, cov, target_return=t, long_only=long_only, bounds=bounds)
        except RuntimeError:
            failed.append(t)          # 不可行目标：跳过，保持已解出点的单调性
            continue
        wv = np.asarray(w, dtype="float64")
        vol = math.sqrt(max(float(wv @ sigma @ wv), 0.0))
        points.append(FrontierPoint(target_return=t, volatility=vol, weights=w))
    if not points:
        raise RuntimeError(f"全部目标收益均不可行：{failed}")
    return Frontier(points=points, failed_targets=tuple(failed))
