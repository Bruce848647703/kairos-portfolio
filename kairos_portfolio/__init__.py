"""Kairos Portfolio —— 自研轻量组合优化与风险度量库。

覆盖组合构建的核心环节：
- moments:  均值/协方差估计（含自研 Ledoit-Wolf 收缩、指数加权协方差）；
- optimize: 等权 / 最小方差 / 最大夏普 / 均值-方差 / 风险平价 / 目标波动；
- risk:     组合波动、Beta、VaR/CVaR、边际与成分风险贡献；
- frontier: 有效前沿；
- realdata: 真实行情加载器（本地日线 CSV -> 统一收盘价面板，离线只读）。

设计原则：解析解优先、必需依赖仅 numpy/pandas（约束优化可选 scipy-SLSQP）、
输出「同型进出」（DataFrame/Series 进 -> Series 出，ndarray 进 -> ndarray 出）。
"""
from .frontier import Frontier, FrontierPoint, efficient_frontier
from .moments import LedoitWolfResult, ema_cov, ledoit_wolf, mean_returns, sample_cov
from .optimize import (
    TargetVolPortfolio,
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_parity,
    target_volatility,
)
from .realdata import load_close_panel
from .risk import (
    RiskContributions,
    beta,
    cvar,
    historical_var,
    norm_ppf,
    parametric_var,
    portfolio_volatility,
    risk_contributions,
)

__version__ = "0.1.0"

__all__ = [
    "mean_returns", "sample_cov", "ledoit_wolf", "LedoitWolfResult", "ema_cov",
    "equal_weight", "min_variance", "max_sharpe", "mean_variance",
    "risk_parity", "target_volatility", "TargetVolPortfolio",
    "portfolio_volatility", "beta", "historical_var", "parametric_var",
    "cvar", "risk_contributions", "RiskContributions", "norm_ppf",
    "efficient_frontier", "Frontier", "FrontierPoint",
    "load_close_panel",
    "__version__",
]
