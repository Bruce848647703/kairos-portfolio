"""realdata.load_close_panel 的离线测试：形状/非正价/停牌 ffill/上市裁剪/确定性。

全部用 tmp_path 造小 CSV，不联网、不读真实数据目录，Python3.9 可复现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos_portfolio.realdata import load_close_panel


def _write_csv(dir_path, symbol, dates, closes, vol=1000.0):
    """写一个 date,open,high,low,close,volume 的日线 CSV（open/high/low 直接复用 close）。"""
    df = pd.DataFrame({
        "date": list(dates),
        "open": list(closes),
        "high": list(closes),
        "low": list(closes),
        "close": list(closes),
        "volume": vol,
    })
    df.to_csv(dir_path / f"{symbol}.csv", index=False)


def _days(start, n):
    return pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")


def test_shape_and_columns_sorted(tmp_path):
    d = _days("2020-01-01", 6)
    _write_csv(tmp_path, "zzz", d, [10, 11, 12, 13, 14, 15])
    _write_csv(tmp_path, "aaa", d, [20, 21, 22, 23, 24, 25])
    panel = load_close_panel(str(tmp_path))
    assert isinstance(panel, pd.DataFrame)
    assert list(panel.columns) == ["aaa", "zzz"]          # 列按代码排序
    assert panel.shape == (6, 2)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.is_monotonic_increasing
    assert panel["aaa"].iloc[0] == pytest.approx(20.0)


def test_non_positive_price_to_nan_then_ffill(tmp_path):
    d = _days("2020-01-01", 5)
    _write_csv(tmp_path, "s1", d, [100.0, 0.0, -5.0, 103.0, 104.0])
    panel = load_close_panel(str(tmp_path))
    col = panel["s1"].to_numpy()
    # 第 2、3 天非正价 -> NaN -> ffill 用第 1 天的 100
    assert col[0] == pytest.approx(100.0)
    assert col[1] == pytest.approx(100.0)
    assert col[2] == pytest.approx(100.0)
    assert col[3] == pytest.approx(103.0)
    assert col[4] == pytest.approx(104.0)


def test_crop_to_common_listing_window(tmp_path):
    _write_csv(tmp_path, "early", _days("2020-01-01", 6), [10, 11, 12, 13, 14, 15])
    _write_csv(tmp_path, "late", _days("2020-01-03", 4), [20, 21, 22, 23])   # 晚上市
    panel = load_close_panel(str(tmp_path))                # drop_incomplete=True
    assert panel.index[0] == pd.Timestamp("2020-01-03")    # 裁剪到全体上市日
    assert panel.index[-1] == pd.Timestamp("2020-01-06")
    assert panel.shape[0] == 4
    assert not panel.isna().any().any()


def test_drop_incomplete_false_keeps_leading_nan(tmp_path):
    _write_csv(tmp_path, "early", _days("2020-01-01", 5), [10, 11, 12, 13, 14])
    _write_csv(tmp_path, "late", _days("2020-01-03", 3), [20, 21, 22])
    panel = load_close_panel(str(tmp_path), drop_incomplete=False)
    assert panel.shape[0] == 5                             # 保留全体日期并集
    assert panel.index[0] == pd.Timestamp("2020-01-01")
    assert np.isnan(panel["late"].iloc[0])                 # 上市前无法 ffill 回填
    assert np.isnan(panel["late"].iloc[1])
    assert panel["late"].iloc[2] == pytest.approx(20.0)


def test_halt_ffill_bridges_middle_gap(tmp_path):
    _write_csv(tmp_path, "early", _days("2020-01-01", 5), [10, 11, 12, 13, 14])
    gap = ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]   # late 在 01-03 停牌
    _write_csv(tmp_path, "late", gap, [20, 21, 22, 23])
    panel = load_close_panel(str(tmp_path))
    assert panel.shape[0] == 5
    row = panel.loc[pd.Timestamp("2020-01-03")]
    assert row["late"] == pytest.approx(21.0)              # ffill 用停牌前 01-02 的 21
    assert row["early"] == pytest.approx(12.0)
    assert not panel.isna().any().any()


def test_determinism_repeated_load_identical(tmp_path):
    d = _days("2020-01-01", 4)
    _write_csv(tmp_path, "a", d, [1, 2, 3, 4])
    _write_csv(tmp_path, "b", d, [5, 6, 7, 8])
    p1 = load_close_panel(str(tmp_path))
    p2 = load_close_panel(str(tmp_path))
    pd.testing.assert_frame_equal(p1, p2)
    assert list(p1.columns) == ["a", "b"]


def test_ignores_non_csv_files(tmp_path):
    _write_csv(tmp_path, "ok", _days("2020-01-01", 3), [1, 2, 3])
    (tmp_path / "DATA_NOTICE.md").write_text("# notice\n", encoding="utf-8")
    panel = load_close_panel(str(tmp_path))
    assert list(panel.columns) == ["ok"]


def test_missing_required_column_raises(tmp_path):
    pd.DataFrame({"date": ["2020-01-01"], "price": [1.0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        load_close_panel(str(tmp_path))


def test_asset_without_any_valid_price_raises(tmp_path):
    d = _days("2020-01-01", 3)
    _write_csv(tmp_path, "good", d, [1, 2, 3])
    _write_csv(tmp_path, "allbad", d, [0, -1, 0])          # 全非正 -> 无有效价
    with pytest.raises(ValueError):
        load_close_panel(str(tmp_path))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_close_panel(str(tmp_path))


def test_missing_dir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_close_panel(str(tmp_path / "nope"))
