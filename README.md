# Kairos Portfolio

> Kairos 量化系列的组合优化模块 —— 一个**自研、轻量**的 Python 组合优化与风险度量库。

`kairos_portfolio` 覆盖「矩估计 → 组合优化 → 风险度量 → 有效前沿」的核心环节：
Ledoit-Wolf 收缩协方差、均值-方差 / 最小方差 / 最大夏普 / 风险平价优化器、
VaR/CVaR 与风险贡献分解。核心代码全部原创，必需依赖仅 `numpy` 与 `pandas`，
含不等式约束的优化可选用 `scipy`（SLSQP）。

## 特性
- **矩估计 `moments`**：简单 / 指数加权均值，样本协方差，**自研 Ledoit-Wolf** 线性收缩
  （对角与常相关两种目标，返回收缩后协方差与收缩强度 δ），RiskMetrics 风格指数加权协方差 `ema_cov`。
- **组合优化 `optimize`**：等权、最小方差、最大夏普（切点组合）、均值-方差（风险厌恶 λ
  或目标收益两种形式）、**自研定点迭代风险平价**（支持自定义风险预算）、目标波动缩放（现金补足）。
- **风险度量 `risk`**：组合波动、Beta、历史 VaR、参数 VaR（自研正态分位数，无需 scipy）、
  CVaR/ES、边际与成分风险贡献分解（可直接校验风险平价）。
- **有效前沿 `frontier`**：在一组目标收益上逐点求解均值-方差，输出 (收益, 波动, 权重) 序列。
- **解析解优先**：无约束最小方差 (w ∝ Σ⁻¹1)、最大夏普 (w ∝ Σ⁻¹(μ−rf·1))、均值-方差
  （KKT 闭式解）均为纯 numpy 计算；long-only / bounds 等约束问题才走 SLSQP。
- **同型进出**：DataFrame/Series 输入返回带资产索引的 Series，纯 ndarray 输入返回 ndarray；
  向量与矩阵成对传入时自动按资产名对齐，杜绝顺序错位。
- **可测试**：测试与示例全部离线、固定 seed 可复现，`pytest -q` 全绿。

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # 或 pip install numpy pandas
pip install -e ".[dev]"       # 需要跑测试时
pip install -e ".[scipy]"     # 需要 long-only/bounds 约束优化时
```

## 快速开始
### ① 矩估计与收缩
```python
import kairos_portfolio as kp

rets = ...  # 收益面板 DataFrame(index=日期, columns=资产)
mu = kp.mean_returns(rets)                       # 或 method="ewm", halflife=20
cov = kp.ledoit_wolf(rets, target="const_corr")  # Ledoit-Wolf 收缩
print(cov.delta)                                 # 收缩强度 δ ∈ [0,1]
cov = cov.cov                                    # 收缩后协方差；也可 kp.sample_cov(rets)
```

### ② 优化与风险分解
```python
w_mv = kp.min_variance(cov, long_only=True)      # 最小方差（SLSQP，需要 scipy）
w_rp = kp.risk_parity(cov)                       # 风险平价（自研定点迭代，纯 numpy）
w_ms = kp.max_sharpe(mu, cov)                    # 最大夏普（无约束解析切点）
rc = kp.risk_contributions(w_rp, cov)
print(rc.pct)                                    # 各资产风险贡献占比 ≈ 1/n
```

### ③ 有效前沿与目标波动
```python
fr = kp.efficient_frontier(kp.mean_returns(rets), cov, n_points=10)
print(fr.frame())                                # target_return / volatility
print(fr.weights_frame())                        # 各点权重

tv = kp.target_volatility(w_mv, cov, target_vol=0.006, max_leverage=1.0)
print(tv.exposure, tv.cash_weight, tv.portfolio_vol)
```

完整可运行示例见 [`examples/demo.py`](examples/demo.py)（long-only 演示需要 scipy，
缺失时自动退化为无约束解析解并给出提示）。

## 真实数据组合优化
除合成数据 demo 外，仓库附**真实行情**的组合优化示例
[`examples/real_portfolio.py`](examples/real_portfolio.py)：加载 `kairos-data` 的本地日线 CSV →
清洗为平衡收盘价面板 → 估计矩（`sample_cov` 与 `ledoit_wolf` 对照，报告收缩强度 δ）→
求等权 / 最小方差 / 最大夏普 / 均值方差 / 风险平价权重 → 计算年化波动、夏普、VaR/CVaR、
成分风险贡献（校验风险平价等风险）与分散化比率 → 生成有效前沿，最终把结果写入
`research/real_portfolio/`（`REPORT.md` 中文报告 + `frontier.csv` + `weights.csv`）。

```bash
# 跨资产 ETF（含国债/黄金/货币，适合风险平价/全天候）
python examples/real_portfolio.py --data-dir /path/to/kairos-data/data/etf
# A 股个股（38 只）
python examples/real_portfolio.py --data-dir /path/to/kairos-data/data/ashare
```

加载器 `kp.load_close_panel(data_dir)` 负责同系列口径：非正价→NaN、ffill 桥接停牌、
按全体上市日裁剪为平衡面板（纯 numpy/pandas、离线只读，`drop_incomplete=False` 可保留并集）。

**示例结论**（跨资产 ETF，2017–2026 公共窗口）：纯风险导向的最小方差 / 风险平价被近零波动的
货币腿捕获（权重 99% / 86%），组合波动极低但收益随之塌缩；样本内最大夏普最高（≈2.0，以债券为主干，
存在前视/过拟合），均值方差（λ=15）给出更可投资的收益—风险平衡（年化收益 ≈10.8%、夏普 ≈1.5），
风险平价分散化比率最高（≈2.1）且成分风险严格等分。完整数字与诚实结论见生成的 `REPORT.md`。

### 数据声明
- 示例所用真实行情来自 `kairos-data` 的**公开行情**（跨资产 ETF / A 股个股，前复权），本仓库只读引用、不再分发。
- 全部结果为历史**样本内**统计与优化演示，期望收益采用已实现均值（含前视偏差），**仅用于研究与学习，不构成任何投资建议**。

## API 概览
| 模块 | 关键对象 | 说明 |
|---|---|---|
| `moments` | `mean_returns` `sample_cov` `ledoit_wolf` `ema_cov` `LedoitWolfResult` | 均值与协方差估计 |
| `optimize` | `equal_weight` `min_variance` `max_sharpe` `mean_variance` `risk_parity` `target_volatility` `TargetVolPortfolio` | 组合优化器 |
| `risk` | `portfolio_volatility` `beta` `historical_var` `parametric_var` `cvar` `risk_contributions` `norm_ppf` | 风险度量 |
| `frontier` | `efficient_frontier` `Frontier` `FrontierPoint` | 有效前沿 |
| `realdata` | `load_close_panel` | 真实行情加载（本地日线 CSV → 平衡收盘价面板，离线只读） |

## 设计要点
- **数据约定**：一切「收益面板」指 DataFrame(index=日期, columns=资产)；协方差 / 期望
  收益也接受纯 ndarray，输出与输入同型。
- **Ledoit-Wolf 收缩**：Σ* = δ·F + (1−δ)·S。δ* 按渐近 oracle 公式 κ̂/n 估计
  （κ̂ = (π̂−ρ̂)/γ̂）；常相关目标的导数修正项 ρ̂ 按 ∂F/∂S 逐项推导成闭式向量化计算，
  并有「数值方向导数」白盒测试交叉验证。传入 `shrinkage=` 可强制指定 δ 做端点对照。
- **风险平价定点迭代**：w ← w ⊙ (b/RC)^damping 后归一化，其中 RC_i = w_i(Σw)_i/σ²；
  从正权重出发迭代恒正，天然 long-only，无需任何求解器；内部按对角几何均值预条件缩放。
- **scipy 缺失时的降级**：解析路径（无约束最小方差 / 最大夏普 / 均值-方差）纯 numpy；
  约束路径抛出带安装指引的中文 `ImportError`，不会静默出错。
- **VaR 符号约定**：VaR / CVaR 一律返回正数损失值；CVaR 取损失超过 VaR 分位的尾部均值，
  恒 ≥ 同置信度 VaR，且对置信度单调不减。

## 测试
```bash
make test          # 或 python -m pytest -q
```

## 项目结构
```
kairos_portfolio/   核心包（moments / optimize / risk / frontier / realdata / _util）
examples/           可运行示例（demo.py 合成数据 / real_portfolio.py 真实行情）
tests/              pytest 测试
research/           真实数据示例产物（real_portfolio/：REPORT.md + frontier.csv + weights.csv，已入库）
```

## 许可
MIT © 2026 Bruce848647703，见 [LICENSE](LICENSE)。

## 参考与致谢
本项目为**独立原创实现**，未复制任何第三方代码。设计思路受业界通用组合优化范式
（均值-方差优化、风险平价、Ledoit-Wolf 协方差收缩、有效前沿、VaR/CVaR 风险度量）
启发，在此向开源量化社区致谢。算法与接口均为本仓库自研。
