"""组合优化模块：等权、最小方差、最大夏普、均值-方差、风险平价、目标波动。

输入约定
--------
- cov: 协方差矩阵，二维 ndarray 或 DataFrame（列名为资产）；
- mu:  期望收益，一维 ndarray 或 Series；当 mu 为 Series 且 cov 为 DataFrame 时
  自动按资产名对齐，避免顺序错位；
- 输出保持「同型进出」：带索引的输入返回 Series，纯数组返回 ndarray。

求解策略
--------
- 解析解优先：无约束最小方差 (w ∝ Σ⁻¹1)、无约束最大夏普（切点 w ∝ Σ⁻¹(μ−rf·1)）、
  无约束均值-方差（KKT 闭式解）均为纯 numpy 计算，不依赖 scipy；
- long-only / bounds 等含不等式约束的问题使用 scipy.optimize.minimize(SLSQP)；
  scipy 缺失时抛出带安装指引的 ImportError，解析路径完全不受影响；
- 风险平价为自研乘性定点迭代，天然 long-only，纯 numpy 实现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._util import Weights, align_pair, as_1d, as_2d, check_psd, solve_psd, wrap_1d

_SLSQP_HINT = (
    "该功能涉及不等式约束（long-only / bounds），需要 SLSQP 数值求解，依赖 scipy。"
    "请先安装：pip install scipy（或 pip install kairos-portfolio[scipy]）。"
)

BoundsLike = Optional[Union[Tuple[float, float], Sequence[Optional[Tuple[float, float]]]]]
_Bounds = Optional[List[Tuple[float, float]]]


def _require_scipy(feature: str):
    """惰性加载 scipy.optimize.minimize；缺失时给出清晰的中文指引。"""
    try:
        from scipy.optimize import minimize

        return minimize
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(f"[{feature}] {_SLSQP_HINT}") from exc


def equal_weight(assets: Union[int, pd.Index, pd.Series, pd.DataFrame, np.ndarray, Sequence]) -> Weights:
    """等权组合。

    参数
    ----
    assets: 资产数量 (int)，或资产索引容器（Index / Series / DataFrame / 字符串数组）。

    返回：int 或数值数组输入 -> ndarray；带资产名的输入 -> Series（index=资产名）。
    """
    if isinstance(assets, (int, np.integer)):
        n = int(assets)
        if n <= 0:
            raise ValueError("资产数量必须为正")
        return np.full(n, 1.0 / n)
    if isinstance(assets, pd.DataFrame):
        cols: pd.Index = assets.columns
    elif isinstance(assets, pd.Series):
        cols = assets.index
    elif isinstance(assets, pd.Index):
        cols = assets
    else:
        arr = np.asarray(list(assets) if not isinstance(assets, np.ndarray) else assets)
        if arr.ndim != 1:
            raise ValueError("assets 必须是 int、一维数组、Index、Series 或 DataFrame")
        if arr.dtype.kind in "USO":
            cols = pd.Index(arr)
        else:
            n = len(arr)
            if n == 0:
                raise ValueError("资产数量必须为正")
            return np.full(n, 1.0 / n)
    n = len(cols)
    if n == 0:
        raise ValueError("资产数量必须为正")
    return pd.Series(np.full(n, 1.0 / n), index=cols, name="weight")


def _resolve_bounds(n: int, long_only: bool, bounds: BoundsLike) -> _Bounds:
    """归一化 bounds 参数。

    - None：由 long_only 决定（True -> 每资产 [0,1]，False -> 无约束）；
    - 单个数值对 (lo, hi)：对全部资产生效；显式给出时不再叠加 long-only 下限；
    - 长度 n 的序列：逐资产 (lo, hi)，元素可为 None 表示该资产无界。
    """
    if bounds is None:
        return [(0.0, 1.0)] * n if long_only else None
    if isinstance(bounds, tuple) and len(bounds) == 2 and all(
        isinstance(v, (int, float, np.floating, np.integer)) for v in bounds
    ):
        pairs = [(float(bounds[0]), float(bounds[1]))] * n
    else:
        seq = list(bounds)
        if len(seq) != n:
            raise ValueError(f"bounds 长度 {len(seq)} 与资产数 {n} 不一致")
        pairs = []
        for b in seq:
            if b is None:
                pairs.append((-np.inf, np.inf))
            else:
                pairs.append((float(b[0]), float(b[1])))
    for lo, hi in pairs:
        if lo > hi:
            raise ValueError(f"bounds 下界 {lo} 大于上界 {hi}")
    return pairs


def _feasible_start(n: int, bnds: _Bounds, w0: Optional[Weights] = None) -> np.ndarray:
    """构造 SLSQP 初始点：优先用户 w0；否则在 bounds 定义的可行域内取权重和为 1 的内点。"""
    if w0 is not None:
        x, _ = as_1d(w0, "w0")
        if x.size != n:
            raise ValueError(f"w0 长度 {x.size} 与资产数 {n} 不一致")
        return x
    if bnds is None:
        return np.full(n, 1.0 / n)
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
        lo_sum = float(lo.sum())
        hi_sum = float(hi.sum())
        if not (lo_sum <= 1.0 + 1e-12 <= hi_sum):
            raise ValueError(f"bounds 无法容纳权重和为 1（[{lo_sum:.4g}, {hi_sum:.4g}]）")
        room = hi - lo
        total = float(room.sum())
        if total <= 0:
            return lo.copy()
        return lo + room * ((1.0 - lo_sum) / total)
    x = np.full(n, 1.0 / n)
    return np.clip(x, lo, hi)


def _sum_constraint(n: int) -> dict:
    """权重和等于 1 的等式约束（带解析梯度）。"""
    return {
        "type": "eq",
        "fun": lambda w: float(w.sum() - 1.0),
        "jac": lambda w: np.ones(n),
    }


def _post(w: np.ndarray, bnds: _Bounds, sum_tol: float = 1e-5) -> np.ndarray:
    """SLSQP 结果后处理：可行性校验 -> 裁剪到 bounds -> 精确归一化到和为 1。"""
    w = np.asarray(w, dtype="float64")
    if not np.all(np.isfinite(w)):
        raise RuntimeError("优化器输出含非有限值，请检查协方差条件数或放宽约束")
    s = float(w.sum())
    if abs(s - 1.0) > sum_tol:
        raise RuntimeError(f"优化未满足权重和为 1 的约束（当前和 {s:.6f}），问题可能不可行")
    if bnds is not None:
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        if np.any(w < lo - 1e-6) or np.any(w > hi + 1e-6):
            raise RuntimeError("优化结果越出 bounds 约束")
        w = np.clip(w, lo, hi)
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    total = float(w.sum())
    if abs(total) < 1e-12:
        raise RuntimeError("归一化失败：权重全为零")
    return w / total


def _slsqp(fun, jac, x0: np.ndarray, bnds: _Bounds, constraints: List[dict],
           feature: str, max_iter: int = 1000) -> np.ndarray:
    """统一的 SLSQP 求解入口（scipy 缺失时在此抛出清晰错误）。"""
    minimize = _require_scipy(feature)
    res = minimize(
        fun, x0, jac=jac, method="SLSQP", bounds=bnds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-14},
    )
    return np.asarray(res.x, dtype="float64")


def min_variance(cov, long_only: bool = True, bounds: BoundsLike = None,
                 w0: Optional[Weights] = None) -> Weights:
    """最小方差组合：min wᵀΣw，s.t. Σw=1（可加 bounds）。

    - long_only=False 且无 bounds：解析解 w ∝ Σ⁻¹1（纯 numpy，允许卖空）；
    - 其余情况：SLSQP 数值求解（需要 scipy）。
    Σ 奇异时解析路径自动退化为伪逆解。
    """
    sigma, index = as_2d(cov, "cov")
    check_psd(sigma, "cov")
    n = sigma.shape[0]
    bnds = _resolve_bounds(n, long_only, bounds)
    if bnds is None:
        w = solve_psd(sigma, np.ones(n))
        s = float(w.sum())
        if not np.isfinite(s) or abs(s) < 1e-12:
            raise ValueError("Σ⁻¹1 的和接近 0，无法归一化；请改用 long_only=True")
        return wrap_1d(w / s, index, name="weight")

    def fun(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (sigma @ w)

    x0 = _feasible_start(n, bnds, w0)
    w = _slsqp(fun, jac, x0, bnds, [_sum_constraint(n)], "min_variance")
    return wrap_1d(_post(w, bnds), index, name="weight")


def max_sharpe(mu, cov, risk_free: float = 0.0, long_only: bool = False,
               bounds: BoundsLike = None, w0: Optional[Weights] = None) -> Weights:
    """最大夏普组合（切点组合）：max (μᵀw − rf) / √(wᵀΣw)，s.t. Σw=1。

    - long_only=False 且无 bounds：解析切点 w ∝ Σ⁻¹(μ − rf·1)，归一化到和为 1。
      注意：当部分资产期望收益低于 rf 时解析解可能含负权重；
    - 其余情况：SLSQP 直接最大化夏普（带解析梯度，需要 scipy）。
    """
    m, sigma, index = align_pair(mu, cov, "mu", "cov")
    check_psd(sigma, "cov")
    n = sigma.shape[0]
    excess = m - float(risk_free)
    bnds = _resolve_bounds(n, long_only, bounds)
    if bnds is None:
        z = solve_psd(sigma, excess)
        s = float(z.sum())
        if not np.isfinite(s) or abs(s) < 1e-12:
            raise ValueError("Σ⁻¹(μ−rf) 的和接近 0，切点组合无定义；请调整 risk_free 或改用 long_only=True")
        return wrap_1d(z / s, index, name="weight")

    def neg_sharpe(w: np.ndarray) -> float:
        var = max(float(w @ sigma @ w), 1e-300)
        return -float(excess @ w) / math.sqrt(var)

    def neg_sharpe_jac(w: np.ndarray) -> np.ndarray:
        var = max(float(w @ sigma @ w), 1e-300)
        vol = math.sqrt(var)
        sw = sigma @ w
        return -(excess * vol - float(excess @ w) * (sw / vol)) / var

    x0 = _feasible_start(n, bnds, w0)
    w = _slsqp(neg_sharpe, neg_sharpe_jac, x0, bnds, [_sum_constraint(n)], "max_sharpe")
    return wrap_1d(_post(w, bnds), index, name="weight")


def mean_variance(mu, cov, risk_aversion: Optional[float] = None,
                  target_return: Optional[float] = None, long_only: bool = True,
                  bounds: BoundsLike = None, w0: Optional[Weights] = None) -> Weights:
    """均值-方差优化，两种形式二选一：

    - risk_aversion=λ：max μᵀw − (λ/2)wᵀΣw，λ 越大越厌恶风险；
    - target_return=r：min wᵀΣw，s.t. μᵀw = r（给定目标收益求最小方差）。

    约束：Σw=1，可叠加 long-only 或自定义 bounds。
    long_only=False 且无 bounds 时使用 KKT 闭式解（纯 numpy）；否则 SLSQP。
    """
    if (risk_aversion is None) == (target_return is None):
        raise ValueError("risk_aversion 与 target_return 必须恰好提供一个")
    m, sigma, index = align_pair(mu, cov, "mu", "cov")
    check_psd(sigma, "cov")
    n = sigma.shape[0]
    one = np.ones(n)
    bnds = _resolve_bounds(n, long_only, bounds)

    if bnds is None:
        # 无约束（除权重和）解析解
        si_one = solve_psd(sigma, one)
        si_mu = solve_psd(sigma, m)
        if risk_aversion is not None:
            lam = float(risk_aversion)
            if lam <= 0:
                raise ValueError("risk_aversion 必须为正")
            a = float(one @ si_one)
            b = float(one @ si_mu)
            nu = (b - lam) / a                     # 权重和约束的拉格朗日乘子
            w = (si_mu - nu * si_one) / lam
            return wrap_1d(w, index, name="weight")
        a = float(one @ si_one)
        b = float(one @ si_mu)
        c = float(m @ si_mu)
        det = a * c - b * b
        if det <= 1e-18 * max(a * c, 1e-300):
            raise ValueError("期望收益退化（μ ∝ 1），目标收益形式无唯一解")
        rt = float(target_return)
        w = ((c - b * rt) * si_one + (a * rt - b) * si_mu) / det
        return wrap_1d(w, index, name="weight")

    def var_fun(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    def var_jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (sigma @ w)

    constraints = [_sum_constraint(n)]
    if risk_aversion is not None:
        lam = float(risk_aversion)
        if lam <= 0:
            raise ValueError("risk_aversion 必须为正")

        def util_fun(w: np.ndarray) -> float:
            return -float(m @ w) + 0.5 * lam * float(w @ sigma @ w)

        def util_jac(w: np.ndarray) -> np.ndarray:
            return -m + lam * (sigma @ w)

        fun, jac = util_fun, util_jac
    else:
        rt = float(target_return)
        fun, jac = var_fun, var_jac
        constraints.append({
            "type": "eq",
            "fun": lambda w: float(m @ w - rt),
            "jac": lambda w: m.copy(),
        })

    x0 = _feasible_start(n, bnds, w0)
    w = _post(_slsqp(fun, jac, x0, bnds, constraints, "mean_variance"), bnds)
    if risk_aversion is None:
        achieved = float(m @ w)
        if abs(achieved - float(target_return)) > 1e-6:
            raise RuntimeError(
                f"目标收益不可行：达到 {achieved:.6f}，目标 {float(target_return):.6f}"
            )
    return wrap_1d(w, index, name="weight")


def risk_parity(cov, risk_budget: Optional[Weights] = None, tol: float = 1e-12,
                max_iter: int = 1000, damping: float = 0.5) -> Weights:
    """风险平价组合（自研乘性定点迭代，纯 numpy）。

    目标：各资产的成分风险贡献占比 RC_i 等于风险预算 b_i（默认 1/n 等分）。
        RC_i = w_i·(Σw)_i / (wᵀΣw)
    迭代：w ← w ⊙ (b/RC)^damping，再归一化权重和为 1。damping ∈ (0,1] 越小越稳、
    收敛越慢。从正权重出发迭代始终为正，天然 long-only，无需求解器。

    内部先按对角线几何均值对 Σ 做预条件缩放（不改变风险贡献占比，改善数值稳定性）。
    """
    sigma, index = as_2d(cov, "cov")
    check_psd(sigma, "cov")
    n = sigma.shape[0]
    d = np.diag(sigma)
    if np.any(d <= 0):
        raise ValueError("风险平价要求各资产方差为正")
    if risk_budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b, _ = as_1d(risk_budget, "risk_budget")
        if b.size != n:
            raise ValueError(f"risk_budget 维度 {b.size} 与资产数 {n} 不一致")
        if np.any(b <= 0):
            raise ValueError("risk_budget 必须全为正")
        b = b / b.sum()

    scale = float(np.exp(np.mean(np.log(d))))      # 预条件：几何均值方差
    s = sigma / scale
    w = b / np.sqrt(d)                             # 逆波动率起点，接近低相关情形的解
    w = w / w.sum()

    converged = False
    for _ in range(int(max_iter)):
        sw = s @ w
        var = float(w @ sw)
        if var <= 0:
            raise RuntimeError("迭代中出现非正组合方差，请检查协方差是否半正定")
        rc = w * sw / var
        if float(np.max(np.abs(rc - b))) <= tol:
            converged = True
            break
        w = w * (b / np.maximum(rc, 1e-300)) ** float(damping)
        w = w / w.sum()
    if not converged:
        sw = s @ w
        rc = w * sw / float(w @ sw)
        gap = float(np.max(np.abs(rc - b)))
        if gap > 1e-7:
            raise RuntimeError(
                f"风险平价迭代未收敛（残差 {gap:.2e}），可增大 max_iter 或减小 damping"
            )
    return wrap_1d(w, index, name="weight")


@dataclass(frozen=True, eq=False)
class TargetVolPortfolio:
    """目标波动缩放结果。

    字段
    ----
    weights:        缩放后的资产权重（总和 = exposure × 原权重和）；
    cash_weight:    现金补足权重 = 1 − exposure × 原权重和；
    exposure:       风险资产总敞口系数（≤ max_leverage）；
    portfolio_vol:  缩放后组合波动（与 cov 同单位，每期）。
    """

    weights: Weights
    cash_weight: float
    exposure: float
    portfolio_vol: float


def target_volatility(weights, cov, target_vol: float,
                      max_leverage: float = 1.0) -> TargetVolPortfolio:
    """按目标波动对既有组合整体缩放，不足部分以现金补足。

    scale = target_vol / σ_p，exposure = min(scale, max_leverage)。
    - 组合波动高于目标：降杠杆（exposure<1，cash_weight>0）；
    - 低于目标且 max_leverage>1：允许有限加杠杆；
    - max_leverage=1 为不杠杆的常见设定（权重和不超过 1）。

    target_vol 与 cov 的量纲必须一致（同为每期，或同为年化）。
    """
    w, sigma, index = align_pair(weights, cov, "weights", "cov")
    check_psd(sigma, "cov")
    tv = float(target_vol)
    if tv <= 0:
        raise ValueError("target_vol 必须为正")
    ml = float(max_leverage)
    if ml <= 0:
        raise ValueError("max_leverage 必须为正")
    var = max(float(w @ sigma @ w), 0.0)
    vol = math.sqrt(var)
    scale = ml if vol <= 1e-14 else tv / vol
    exposure = min(scale, ml)
    invested = exposure * float(w.sum())
    return TargetVolPortfolio(
        weights=wrap_1d(exposure * w, index, name="weight"),
        cash_weight=1.0 - invested,
        exposure=exposure,
        portfolio_vol=exposure * vol,
    )
