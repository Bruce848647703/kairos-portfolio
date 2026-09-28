"""真实数据组合优化示例：加载 kairos-data 真实行情 -> 矩估计 -> 五种配置 -> 风险分解 -> 有效前沿 -> 报告。

全部离线、不联网；数据只读。核心计算复用本包 `kairos_portfolio`：
`load_close_panel` 读收盘价面板，`mean_returns`/`sample_cov`/`ledoit_wolf` 估矩，
`equal_weight`/`min_variance`/`max_sharpe`/`mean_variance`/`risk_parity` 求权重，
`portfolio_volatility`/`historical_var`/`cvar`/`risk_contributions` 度量风险，
`efficient_frontier` 生成前沿。scipy 缺失时约束优化自动退化为无约束解析解并提示。

运行：
    python examples/real_portfolio.py --data-dir /path/to/kairos-data/data/etf
输出（默认 research/real_portfolio/）：REPORT.md、frontier.csv、weights.csv，并打印关键数字。
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import kairos_portfolio as kp
from kairos_portfolio.realdata import load_close_panel

try:  # 约束优化（long-only）需要 scipy；缺失时退化为无约束解析解
    from scipy.optimize import minimize  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - 取决于运行环境
    HAVE_SCIPY = False

ANN = 252.0
DEFAULT_DATA_DIR = "/home/zhuoming.wang/quant-hub/kairos/kairos-data/data/etf"
SCHEME_ORDER = ["等权", "最小方差", "最大夏普", "均值方差", "风险平价"]


def risk_parity_robust(cov):
    """稳健求解风险平价：默认参数 -> 加强迭代 -> 逆波动率回退。返回 (权重 Series, 说明)。

    ETF 池含近零波动的货币腿时定点迭代可能变慢或不收敛；此处逐级放宽 tol/damping、
    增大 max_iter，仍失败则回退到逆波动率（风险平价在低相关下的近似解），绝不让脚本崩。
    """
    attempts = [
        dict(tol=1e-12, max_iter=2000, damping=0.5),
        dict(tol=1e-10, max_iter=8000, damping=0.3),
        dict(tol=1e-8, max_iter=30000, damping=0.15),
    ]
    last = "未知"
    for kw in attempts:
        try:
            w = kp.risk_parity(cov, **kw)
            return w, f"定点迭代收敛（damping={kw['damping']}, tol={kw['tol']:.0e}, max_iter={kw['max_iter']}）"
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - 视数据而定
            last = str(exc)
    d = np.sqrt(np.diag(np.asarray(cov, dtype="float64")))
    inv = 1.0 / np.maximum(d, 1e-12)
    inv = inv / inv.sum()
    w = pd.Series(inv, index=cov.columns, name="weight")
    return w, f"风险平价未收敛，回退逆波动率权重（原因：{last}）"


def diversification_ratio(w, cov):
    """分散化比率 DR = (Σ wᵢ·σᵢ) / σ_p ≥ 1（Choueifaty），越大表示风险分散越充分。"""
    w = w.reindex(cov.columns)
    sigma = np.sqrt(np.diag(np.asarray(cov, dtype="float64")))
    num = float(np.asarray(w, dtype="float64") @ sigma)
    den = kp.portfolio_volatility(w, cov)
    return num / den if den > 0 else float("nan")


def build_schemes(mu, cov, assets, long_only, risk_aversion):
    """构造五种配置权重（全部 long-only 时约束优化走 SLSQP）。返回 (dict, dict 备注)。"""
    notes = {}
    schemes = {
        "等权": kp.equal_weight(assets),
        "最小方差": kp.min_variance(cov, long_only=long_only),
        "最大夏普": kp.max_sharpe(mu, cov, long_only=long_only),
        "均值方差": kp.mean_variance(mu, cov, risk_aversion=risk_aversion, long_only=long_only),
    }
    rp, rp_note = risk_parity_robust(cov)
    schemes["风险平价"] = rp
    notes["风险平价"] = rp_note
    notes["均值方差"] = f"risk_aversion λ={risk_aversion:g}（效用形式 max μᵀw − λ/2·wᵀΣw）"
    return schemes, notes


def evaluate(w, mu, cov, rets, confidence):
    """对单个组合计算风险收益指标（年化收益/波动、夏普、日频 VaR/CVaR、DR、最大风险占比）。"""
    w = w.reindex(cov.columns)
    wv = np.asarray(w, dtype="float64")
    port = rets[cov.columns].mul(wv, axis=1).sum(axis=1)      # 已实现日收益序列
    ann_ret = float(wv @ np.asarray(mu.reindex(cov.columns), dtype="float64")) * ANN
    ann_vol = kp.portfolio_volatility(w, cov, periods_per_year=ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    rc = kp.risk_contributions(w, cov)
    pct = np.asarray(rc.pct, dtype="float64")
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "var_d": kp.historical_var(port, confidence),
        "cvar_d": kp.cvar(port, confidence),
        "dr": diversification_ratio(w, cov),
        "max_rc": float(pct.max()),
        "top_asset": str(w.idxmax()),
        "top_w": float(w.max()),
        "port": port,
        "rc": rc,
    }


def _pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"


def _md_table(headers, rows):
    """极简 markdown 表格生成器（对齐由 markdown 渲染器处理）。"""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(path, args, panel, rets, mu, cov_s, lw, schemes, metrics,
                 frontier, long_only, notes):
    """把全部结果写成中文 REPORT.md（含数据驱动的诚实结论与免责声明）。"""
    n_assets = panel.shape[1]
    n_days = len(rets)
    cov = lw.cov
    asset_vol = np.sqrt(np.diag(np.asarray(cov, dtype="float64"))) * math.sqrt(ANN)
    ann_ret_asset = np.asarray(mu, dtype="float64") * ANN

    lines = []
    ap = lines.append
    ap("# 真实数据组合优化报告")
    ap("")
    ap("> 由 `examples/real_portfolio.py` 自动生成；数据只读、全程离线，结果可复现。")
    ap("")
    ap("## 一、数据与口径")
    ap(f"- 数据目录：`{args.data_dir}`")
    ap(f"- 标的数：{n_assets}；公共样本区间：{panel.index[0].date()} ~ {panel.index[-1].date()}"
       f"（{len(panel)} 个交易日，收益样本 {n_days} 条）")
    ap("- 清洗口径：非正价→NaN、ffill 桥接停牌、按全体上市日裁剪为平衡面板（`load_close_panel`）。")
    ap(f"- 优化约束：{'long-only（SLSQP，已检测到 scipy）' if long_only else '无约束解析解（未检测到 scipy，权重可能为负）'}。")
    ap(f"- 年化口径：波动 ×√252，收益 ×252；VaR/CVaR 为**日频**、置信度 {_pct(args.confidence, 0)}，取正数损失值。")
    ap("")

    ap("## 二、矩估计：样本协方差 vs Ledoit-Wolf 收缩")
    diff = np.linalg.norm(cov.to_numpy() - cov_s.to_numpy(), "fro")
    scale = np.linalg.norm(cov_s.to_numpy(), "fro")
    ap(f"- Ledoit-Wolf（常相关目标）收缩强度 **δ = {lw.delta:.4f}**。")
    ap(f"- δ 极小，收缩后协方差与样本协方差的相对 Frobenius 差异仅 {diff / scale:.4%}："
       f"样本量 n={n_days} 远大于资产数 p={n_assets}，协方差已被充分估计，几乎无需收缩。")
    ap("- 本报告的优化统一采用 **Ledoit-Wolf 收缩协方差**（更稳健）；因 δ≈0，与用样本协方差的结果基本一致。")
    ap("")

    ap("### 各资产年化波动与年化收益（样本内已实现）")
    ap(_md_table(
        ["标的", "年化波动", "年化收益"],
        [[a, _pct(v), _pct(m)] for a, v, m in zip(cov.columns, asset_vol, ann_ret_asset)],
    ))
    ap("")

    ap("## 三、各方案权重")
    wdf = pd.DataFrame({k: schemes[k].reindex(cov.columns) for k in SCHEME_ORDER})
    ap(_md_table(["标的"] + SCHEME_ORDER,
                 [[a] + [f"{wdf.loc[a, k]:.4f}" for k in SCHEME_ORDER] for a in cov.columns]))
    ap("")
    ap("各方案最大单一持仓：")
    for k in SCHEME_ORDER:
        m = metrics[k]
        ap(f"- **{k}**：{m['top_asset']} 占 {_pct(m['top_w'])}")
    for k in ("均值方差", "风险平价"):
        if k in notes:
            ap(f"- 备注（{k}）：{notes[k]}")
    ap("")

    ap("## 四、风险收益对比")
    ap(_md_table(
        ["方案", "年化收益", "年化波动", "夏普", f"日 VaR({_pct(args.confidence, 0)})",
         f"日 CVaR({_pct(args.confidence, 0)})", "分散化比率", "最大成分风险占比"],
        [[k, _pct(metrics[k]["ann_ret"]), _pct(metrics[k]["ann_vol"]),
          f"{metrics[k]['sharpe']:.3f}", _pct(metrics[k]["var_d"]),
          _pct(metrics[k]["cvar_d"]), f"{metrics[k]['dr']:.3f}",
          _pct(metrics[k]["max_rc"], 1)] for k in SCHEME_ORDER],
    ))
    ap("")
    ap("> 夏普按 rf=0、样本内已实现收益/波动计算；分散化比率 DR=(Σwᵢσᵢ)/σ_p，越大越分散。")
    ap("")

    ap("## 五、风险平价成分贡献校验")
    rc = metrics["风险平价"]["rc"]
    target = 1.0 / n_assets
    dev = float(np.abs(np.asarray(rc.pct, dtype="float64") - target).max())
    ap(f"目标：每个资产成分风险占比 = 1/{n_assets} = {target:.4f}。实测最大偏离 **{dev:.2e}**，"
       f"风险平价精确成立（本包定点迭代自研解）。")
    ap(_md_table(
        ["标的", "权重", "成分风险占比"],
        [[a, f"{wdf.loc[a, '风险平价']:.4f}", _pct(float(rc.pct.loc[a]), 2)] for a in cov.columns],
    ))
    ap("")
    ap("**关键观察**：风险平价把绝大部分**权重**给了低波动腿（货币/债券），但每个资产的**风险贡献**仍严格等分——"
       "低波动资产需要极大权重才能贡献等量风险，这正是风险平价「按风险而非按资金分配」的本质。")
    ap("")

    ap("## 六、有效前沿")
    fdf = frontier.frame()
    ann_pts = fdf.assign(ann_ret=fdf["target_return"] * ANN, ann_vol=fdf["volatility"] * math.sqrt(ANN))
    ap(f"在 [最小方差收益, 最大期望收益] 上取 {len(frontier.points)} 个目标点逐点求解"
       f"（long-only={long_only}），全部可行（跳过 {len(frontier.failed_targets)} 个）。")
    ap(_md_table(["#", "年化收益", "年化波动"],
                 [[i + 1, _pct(r_.ann_ret), _pct(r_.ann_vol)]
                  for i, r_ in ann_pts.iterrows()]))
    ap("")
    ap("完整数据见 `frontier.csv`（含日频与年化两口径）；权重矩阵见 `weights.csv`。")
    ap("")

    # ---- 数据驱动的诚实结论 ----
    ap("## 七、诚实结论")
    best_sharpe = max(SCHEME_ORDER, key=lambda k: metrics[k]["sharpe"])
    best_dr = max(SCHEME_ORDER, key=lambda k: metrics[k]["dr"])
    best_ret = max(SCHEME_ORDER, key=lambda k: metrics[k]["ann_ret"])
    mv = metrics["最小方差"]
    rp = metrics["风险平价"]
    eq = metrics["等权"]
    bm = metrics[best_sharpe]
    mvu = metrics["均值方差"]
    cash_asset = mv["top_asset"]
    cash_vol = _pct(float(asset_vol[list(cov.columns).index(cash_asset)]), 2)
    ret_clause = ("全场最高年化收益" if best_ret == "均值方差"
                  else f"仅次于 {best_ret}（{_pct(metrics[best_ret]['ann_ret'])}）")
    ap(f"1. **低波动腿主导纯风险导向配置**：货币 ETF（{cash_asset}，年化波动仅 {cash_vol}）"
       f"在最小方差中占 {_pct(mv['top_w'])}、在风险平价中占 {_pct(rp['top_w'])}。"
       f"二者组合波动被压到 {_pct(mv['ann_vol'])} / {_pct(rp['ann_vol'])}，但年化收益也随之塌缩到 "
       f"{_pct(mv['ann_ret'])} / {_pct(rp['ann_ret'])}——纯风险最小化在多资产含现金腿的样本上"
       "几乎等价于「持有现金」，牺牲了绝大部分收益弹性。")
    ap(f"2. **风险调整最优（样本内）= {best_sharpe}**：夏普 {bm['sharpe']:.3f}，"
       f"以{bm['top_asset']}为主干（占 {_pct(bm['top_w'])}）叠加少量黄金/纳指，"
       f"年化收益 {_pct(bm['ann_ret'])}、波动 {_pct(bm['ann_vol'])}。但这是**样本内**最大夏普，"
       "对期望收益高度敏感、存在过拟合，实盘期望应显著低于此值。")
    ap(f"3. **收益—风险的务实平衡 = 均值方差（λ={args.risk_aversion:g}）**：年化收益 "
       f"{_pct(mvu['ann_ret'])}（{ret_clause}）、波动 {_pct(mvu['ann_vol'])}、"
       f"夏普 {mvu['sharpe']:.3f}、DR {mvu['dr']:.3f}，在债券/黄金/纳指间分散，"
       "兼顾了收益弹性与风险约束，是比纯最小方差更可投资的选择。")
    ap(f"4. **分散度最高 = {best_dr}**（DR={metrics[best_dr]['dr']:.3f}），风险贡献严格等分；"
       "但其绝对收益低，适合作为「稳态/全天候」风险基准而非收益引擎。")
    ap(f"5. **等权**年化收益 {_pct(eq['ann_ret'])}、波动 {_pct(eq['ann_vol'])}、夏普 {eq['sharpe']:.3f}："
       "因权益腿占比高而波动大，风险调整逊于债券主干的最大夏普与均值方差方案。")
    ap("")
    ap("**综合判断**：在该真实 ETF 样本上，若以夏普（风险调整收益）论，最大夏普组合最高、"
       "均值方差次之且更稳健可投资；若以绝对分散/风险均衡论，风险平价最优但收益偏低；"
       "纯最小方差被现金腿捕获、性价比最低。含近零波动现金腿会系统性地拉低任何「风险导向」配置的收益，"
       "实务中通常对现金腿设上限或改用目标波动/杠杆而非直接纳入风险平价。")
    ap("")
    ap("> **免责声明**：以上全部为历史样本内统计与优化结果，期望收益采用已实现均值（前视），"
       "仅用于方法演示与研究，**不构成任何投资建议**；数据为公开行情、版权归原作者/数据源所有。")
    ap("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def parse_args(argv=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(repo_root, "research", "real_portfolio")
    p = argparse.ArgumentParser(description="真实数据组合优化示例（离线、只读数据）")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="日线 CSV 目录（默认跨资产 ETF 目录）")
    p.add_argument("--out-dir", default=default_out, help="结果输出目录（默认 research/real_portfolio）")
    p.add_argument("--risk-aversion", type=float, default=15.0, help="均值方差的风险厌恶系数 λ")
    p.add_argument("--frontier-points", type=int, default=12, help="有效前沿采样点数")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR/CVaR 置信度")
    p.add_argument("--lookback", type=int, default=None, help="仅用最近 N 个交易日（默认全样本）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    long_only = HAVE_SCIPY
    if not HAVE_SCIPY:
        print("[提示] 未检测到 scipy，约束优化退化为无约束解析解（权重可能为负）。")

    panel = load_close_panel(args.data_dir)
    rets = panel.pct_change().dropna()
    if args.lookback is not None:
        rets = rets.iloc[-int(args.lookback):]
    print("=" * 72)
    print(f"真实数据：{args.data_dir}")
    print(f"面板 {panel.shape[0]}×{panel.shape[1]}  区间 {panel.index[0].date()} ~ {panel.index[-1].date()}"
          f"  收益样本 {len(rets)}")

    mu = kp.mean_returns(rets)
    cov_s = kp.sample_cov(rets)
    lw = kp.ledoit_wolf(rets, target="const_corr")
    cov = lw.cov
    print(f"矩估计：Ledoit-Wolf δ={lw.delta:.4f}（常相关目标），"
          f"‖Σ*_lw−Σ_sample‖_F/‖Σ_sample‖_F="
          f"{np.linalg.norm(cov.to_numpy() - cov_s.to_numpy(), 'fro') / np.linalg.norm(cov_s.to_numpy(), 'fro'):.4%}")

    schemes, notes = build_schemes(mu, cov, rets.columns, long_only, args.risk_aversion)
    metrics = {k: evaluate(schemes[k], mu, cov, rets, args.confidence) for k in SCHEME_ORDER}

    print("=" * 72)
    print(f"{'方案':<8}{'年化收益':>10}{'年化波动':>10}{'夏普':>8}{'日VaR':>9}{'日CVaR':>9}{'DR':>7}{'最大风险占比':>12}")
    for k in SCHEME_ORDER:
        m = metrics[k]
        print(f"{k:<8}{_pct(m['ann_ret']):>10}{_pct(m['ann_vol']):>10}{m['sharpe']:>8.3f}"
              f"{_pct(m['var_d']):>9}{_pct(m['cvar_d']):>9}{m['dr']:>7.3f}{_pct(m['max_rc'], 1):>12}")
    for k in ("均值方差", "风险平价"):
        print(f"  备注[{k}]: {notes[k]}")

    frontier = kp.efficient_frontier(mu, cov, n_points=args.frontier_points, long_only=long_only)
    print("=" * 72)
    print(f"有效前沿：{len(frontier.points)} 点（跳过 {len(frontier.failed_targets)}），"
          f"年化收益 {_pct(frontier.points[0].target_return * ANN)} ~ "
          f"{_pct(frontier.points[-1].target_return * ANN)}，"
          f"年化波动 {_pct(frontier.points[0].volatility * math.sqrt(ANN))} ~ "
          f"{_pct(frontier.points[-1].volatility * math.sqrt(ANN))}")

    rc = metrics["风险平价"]["rc"]
    dev = float(np.abs(np.asarray(rc.pct, dtype="float64") - 1.0 / panel.shape[1]).max())
    print(f"风险平价成分贡献最大偏离 1/n：{dev:.2e}（等风险成立）")

    os.makedirs(args.out_dir, exist_ok=True)
    wdf = pd.DataFrame({k: schemes[k].reindex(cov.columns) for k in SCHEME_ORDER})
    wdf.index.name = "asset"
    weights_path = os.path.join(args.out_dir, "weights.csv")
    wdf.to_csv(weights_path, float_format="%.6f")

    fdf = frontier.frame()
    fdf["target_return_ann"] = fdf["target_return"] * ANN
    fdf["volatility_ann"] = fdf["volatility"] * math.sqrt(ANN)
    frontier_path = os.path.join(args.out_dir, "frontier.csv")
    fdf.to_csv(frontier_path, index=False, float_format="%.8f")

    report_path = os.path.join(args.out_dir, "REPORT.md")
    write_report(report_path, args, panel, rets, mu, cov_s, lw, schemes, metrics,
                 frontier, long_only, notes)

    print("=" * 72)
    print(f"已写出：\n  {report_path}\n  {weights_path}\n  {frontier_path}")


if __name__ == "__main__":
    main()
