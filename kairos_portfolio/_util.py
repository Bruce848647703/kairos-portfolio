"""内部通用工具：输入形状归一、对称 PSD 线性代数辅助、输出类型包装。

不属于公开 API，供 moments / optimize / risk / frontier 复用。
核心约定是「同型进出」：带 pandas 索引的输入返回 Series，纯 ndarray 输入返回 ndarray。
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

Weights = Union[np.ndarray, pd.Series]


def as_2d(cov, name: str = "cov") -> Tuple[np.ndarray, Optional[pd.Index]]:
    """把协方差类输入归一为 (对称化 2 维 ndarray, 列索引或 None)。

    DataFrame 输入会携带列索引，用于输出时还原成 Series；并做对称化处理，
    消除浮点噪声导致的微小不对称。
    """
    if isinstance(cov, pd.DataFrame):
        arr = cov.to_numpy(dtype="float64")
        index: Optional[pd.Index] = cov.columns
    else:
        arr = np.asarray(cov, dtype="float64")
        index = None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} 必须是方阵，当前形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/Inf")
    return (arr + arr.T) / 2.0, index


def as_1d(x, name: str = "x") -> Tuple[np.ndarray, Optional[pd.Index]]:
    """把向量类输入归一为 (1 维 float ndarray, 索引或 None)。"""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype="float64")
        index: Optional[pd.Index] = x.index
    else:
        arr = np.asarray(x, dtype="float64").ravel()
        index = None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/Inf")
    return arr, index


def align_pair(vec, mat, vec_name: str = "x",
               mat_name: str = "cov") -> Tuple[np.ndarray, np.ndarray, Optional[pd.Index]]:
    """归一化「(向量, 矩阵)」成对输入。

    当矩阵是 DataFrame 且向量是 Series 时，先按矩阵列名 reindex 对齐，
    避免资产顺序不同造成静默错位。返回 (向量, 矩阵, 输出用索引)。
    """
    a, mat_index = as_2d(mat, mat_name)
    if mat_index is not None and isinstance(vec, pd.Series):
        v, _ = as_1d(vec.reindex(mat_index), vec_name)
        index = mat_index
    else:
        v, vec_index = as_1d(vec, vec_name)
        index = mat_index if mat_index is not None else vec_index
    if v.size != a.shape[0]:
        raise ValueError(f"{vec_name} 维度 {v.size} 与 {mat_name} 维度 {a.shape[0]} 不一致")
    return v, a, index


def wrap_1d(values: np.ndarray, index: Optional[pd.Index] = None,
            name: Optional[str] = None) -> Weights:
    """有索引输出 Series，否则输出 ndarray，保持「同型进出」。"""
    arr = np.asarray(values, dtype="float64")
    if index is not None:
        return pd.Series(arr, index=index, name=name)
    return arr


def solve_psd(a: np.ndarray, b: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """解线性方程组 a·x = b；矩阵奇异或病态时退化为伪逆，保证优雅降级。"""
    try:
        x = np.linalg.solve(a, b)
        if np.all(np.isfinite(x)):
            return x
    except np.linalg.LinAlgError:
        pass
    scale = max(float(np.trace(a)) / max(a.shape[0], 1), 1e-16)
    return np.linalg.pinv(a + ridge * scale * np.eye(a.shape[0])) @ b


def check_psd(a: np.ndarray, name: str = "cov") -> None:
    """轻量校验矩阵近似半正定（允许浮点噪声级别的微小负特征值）。"""
    eig = np.linalg.eigvalsh(a)
    floor = -1e-10 * max(1.0, float(np.abs(eig).max()))
    if float(eig.min()) < floor:
        raise ValueError(f"{name} 必须半正定，最小特征值 {float(eig.min()):.3e}")
