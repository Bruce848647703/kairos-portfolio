"""Kairos Portfolio 演示：矩估计 -> 四种优化器 -> 风险度量 -> 有效前沿 -> 目标波动。

运行： python examples/demo.py
所有数据均为离线合成（固定 seed），可复现；未安装 scipy 时自动退化为无约束解析解。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_portfolio as kp

try:  # 约束优化（long-only）需要 scipy；缺失时演示自动退化为无约束解析解
    from scipy.optimize import minimize  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def make_synthetic_returns(n: int = 1008, seed: int = 3) -> pd.DataFrame:
    """合成 4 资产日收益：2 个公共因子（类股票/债券驱动）+ 特质噪声 + 漂移。"""
    rng = np.random.default_rng(seed)
    names = ["STOCK_A", "STOCK_B", "BOND_C", "GOLD_D"]
    load = np.array([
        [1.10, 0.20],   # STOCK_A: 重仓股票因子
        [0.90, 0.35],   # STOCK_B: 股票因子为主
        [0.15, 0.80],   # BOND_C:  债券因子为主
        [0.30, -0.60],  # GOLD_D:  与债券因子负相关（对冲属性）
    ])
    f = rng.standard_normal((n, 2)) * np.array([0.010, 0.006])
    idio = rng.standard_normal((n, 4)) * np.array([0.008, 0.010, 0.003, 0.007])
    drift = np.array([0.00035, 0.00028, 0.00009, 0.00012])
    rets = f @ load.T + idio + drift
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(rets, index=idx, columns=names)


def main() -> None:
    ann = math.sqrt(252.0)
    rets = make_synthetic_returns()
    long_only = HAVE_SCIPY
    if not HAVE_SCIPY:
        print("[提示] 未检测到 scipy，约束优化退化为无约束解析解（权重可能为负）。")

    print("=" * 66)
    print("① 矩估计：样本 vs Ledoit-Wolf 收缩（n=%d, %d 资产）" % rets.shape)
    mu = kp.mean_returns(rets)
    cov = kp.sample_cov(rets)
    lw = kp.ledoit_wolf(rets, target="const_corr")
    d = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(d, d)
    off = corr[~np.eye(len(corr), dtype=bool)]
    print("日均收益:", {k: round(v, 5) for k, v in mu.items()})
    print(f"样本相关系数（非对角）范围: {off.min():+.3f} ~ {off.max():+.3f}")
    print(f"Ledoit-Wolf 收缩强度 delta = {lw.delta:.4f}（目标: 常相关）")

    print("=" * 66)
    tag = "long-only" if long_only else "无约束解析"
    print(f"② 优化器对比：等权 / 最小方差 / 风险平价 / 最大夏普（{tag}）")
    schemes = {
        "等权": kp.equal_weight(rets.columns),
        "最小方差": kp.min_variance(cov, long_only=long_only),
        "风险平价": kp.risk_parity(cov),
        "最大夏普": kp.max_sharpe(mu, cov, long_only=long_only),
    }
    print("权重:")
    print(pd.DataFrame(schemes).round(4).to_string())
    rows = {}
    for name, w in schemes.items():
        wv = np.asarray(w, dtype="float64")
        rc = kp.risk_contributions(w, cov)
        port = rets.mul(w, axis=1).sum(axis=1)
        ret_ann = float(wv @ mu.to_numpy()) * 252
        vol_ann = rc.volatility * ann
        rows[name] = {
            "年化收益": ret_ann,
            "年化波动": vol_ann,
            "夏普": ret_ann / vol_ann if vol_ann > 0 else float("nan"),
            "95%VaR(日)": kp.historical_var(port, 0.95),
            "95%CVaR(日)": kp.cvar(port, 0.95),
            "最大单资产风险占比": float(np.asarray(rc.pct).max()),
        }
    print("风险收益对比:")
    print(pd.DataFrame(rows).T.round(4).to_string())

    print("=" * 66)
    print("③ 有效前沿（7 个目标收益，年化口径）")
    fr = kp.efficient_frontier(mu, cov, n_points=7, long_only=long_only)
    fdf = fr.frame()
    fdf["年化收益"] = fdf["target_return"] * 252
    fdf["年化波动"] = fdf["volatility"] * ann
    print(fdf[["年化收益", "年化波动"]].round(4).to_string(index=False))
    if fr.failed_targets:
        print(f"(跳过不可行目标: {fr.failed_targets})")

    print("=" * 66)
    print("④ 目标波动缩放：把最大夏普组合缩放到年化 10% 波动，其余现金补足")
    tv = kp.target_volatility(schemes["最大夏普"], cov, target_vol=0.10 / ann, max_leverage=1.0)
    print(f"总敞口={tv.exposure:.2%}  现金权重={tv.cash_weight:.2%}  "
          f"缩放后年化波动={tv.portfolio_vol * ann:.2%}")


if __name__ == "__main__":
    main()
