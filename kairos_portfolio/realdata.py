"""真实行情加载器：把本地日线 CSV 目录读成统一的收盘价面板。

数据约定（与 kairos-data 对齐）：目录内每个 ``*.csv`` 为一个标的的日线，含
``date,open,high,low,close,volume`` 六列；文件名去扩展名即标的代码（如 ``sh510300``）。
本模块仅依赖 numpy/pandas，**离线读取、不联网**，非 CSV 文件（如 DATA_NOTICE.md）自动忽略。

同系列口径（保证跨标的可比、结果确定）：

1. 非正价（``close<=0`` 或缺失/非数值）一律置为 NaN —— 视作停牌或异常报价；
2. 前向填充 ``ffill`` 桥接停牌/缺交易的日期（领先缺失即上市前无价，无法回填，保留 NaN）；
3. 按「全体上市日」裁剪：``drop_incomplete=True``（默认）时只保留所有标的都已上市的
   公共区间 ``[max(各标的首个有效日), min(各标的末个有效日)]``，得到平衡面板。

列按标的代码排序，日期升序，同一目录多次加载结果完全一致（确定性）。
"""
from __future__ import annotations

import os
from glob import glob
from typing import List

import pandas as pd

__all__ = ["load_close_panel"]

_REQUIRED = ("date", "close")


def _read_one_close(path: str) -> pd.Series:
    """读取单个日线 CSV 的收盘价。

    返回 ``index=DatetimeIndex``（升序、去重）、``name=标的代码`` 的 Series；
    非正价与无法解析为数值的价格均置 NaN。
    """
    symbol = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)
    lower = {str(c).lower().strip(): c for c in df.columns}
    missing = [k for k in _REQUIRED if k not in lower]
    if missing:
        raise ValueError(f"{path} 缺少必需列 {missing}，实际列 {list(df.columns)}")
    close = pd.to_numeric(df[lower["close"]], errors="coerce").to_numpy(dtype="float64")
    s = pd.Series(close, index=pd.to_datetime(df[lower["date"]]), name=symbol)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.where(s > 0.0)          # 非正价 -> NaN（停牌/异常报价）


def load_close_panel(data_dir: str, drop_incomplete: bool = True) -> pd.DataFrame:
    """加载目录下全部 ``*.csv`` 的收盘价，拼成统一面板。

    参数
    ----
    data_dir:        日线 CSV 所在目录（只读；非 CSV 文件忽略）。
    drop_incomplete: True（默认）裁剪到「全体上市日」的公共区间，返回无缺失的平衡面板；
                     False 保留全体日期并集（上市前为 NaN），仅做 ffill。

    返回：``DataFrame(index=日期升序, columns=标的代码升序, values=收盘价)``。

    异常：目录不存在 -> ``NotADirectoryError``；无 CSV -> ``FileNotFoundError``；
    某标的无任何有效正价、或公共区间为空 -> ``ValueError``。
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"数据目录不存在：{data_dir}")
    paths = sorted(glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"目录下没有 CSV 文件：{data_dir}")

    series: List[pd.Series] = [_read_one_close(p) for p in paths]
    panel = pd.concat(series, axis=1).sort_index()
    panel = panel[sorted(panel.columns)]            # 列按代码排序，保证确定性

    dead = list(panel.columns[panel.isna().all(axis=0)])
    if dead:
        raise ValueError(f"以下标的无任何有效正收盘价，无法纳入面板：{dead}")

    panel = panel.ffill()                            # 桥接停牌/缺交易的中间日期
    if not drop_incomplete:
        return panel

    raw = pd.concat(series, axis=1).sort_index()[sorted(panel.columns)]
    first_valid = raw.apply(lambda s: s.first_valid_index())
    last_valid = raw.apply(lambda s: s.last_valid_index())
    start, end = first_valid.max(), last_valid.min()  # 全体上市公共区间
    if end < start:
        raise ValueError("各标的无公共交易日区间（上市/退市时间不重叠）")
    panel = panel.loc[start:end].dropna(how="any")    # dropna 为安全兜底
    if panel.empty:
        raise ValueError("裁剪后收盘面板为空，请检查各标的是否有公共交易日")
    return panel
