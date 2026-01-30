# -*- coding: utf-8 -*-
"""
股票技術形態辨識系統 (Stock Pattern Recognition System)
支援形態: 頭肩頂/底、雙頂/底、三角收斂、上升/下降通道
包含 12 小時數據快取機制
"""

# ============================================================
# 套件匯入 (Library Imports)
# ============================================================

# 標準函式庫
import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# 數據處理
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress

# 股票數據
import yfinance as yf

# 繪圖
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplfinance as mpf

# 設定中文字體 (Windows: Microsoft JhengHei, macOS: PingFang TC, Linux: Noto Sans CJK TC)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


# ============================================================
# 快取設定常數 (Cache Configuration)
# ============================================================

# 快取有效時間：12 小時 (單位：秒)
CACHE_EXPIRY_HOURS = 12
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_HOURS * 60 * 60

# 快取目錄路徑
CACHE_DIR = Path(__file__).parent / ".cache"


# ============================================================
# 資料結構定義 (Data Structures)
# ============================================================

@dataclass
class Pattern:
    """技術形態資料結構"""
    name: str                           # 形態名稱
    start_idx: int                      # 起始索引
    end_idx: int                        # 結束索引
    key_points: List[Tuple[int, float]] # 關鍵點 (索引, 價格)
    pattern_type: str                   # 'bullish' 或 'bearish'
    confidence: float                   # 信心度 0-1


# ============================================================
# 快取管理模組 (Cache Management Module)
# ============================================================

class DataCache:
    """
    股票數據快取管理器

    實作 12 小時快取機制，避免頻繁呼叫 API：
    - 快取檔案以 symbol_period_interval 的 hash 值命名
    - 每次讀取時檢查快取時間戳記
    - 超過 12 小時自動重新下載
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        """
        初始化快取管理器

        Args:
            cache_dir: 快取目錄路徑
        """
        self.cache_dir = cache_dir
        # 確保快取目錄存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """
        產生快取鍵值

        使用 MD5 雜湊產生唯一識別碼，避免檔名過長或含有特殊字元

        Args:
            symbol: 股票代號
            period: 資料期間
            interval: K線週期

        Returns:
            快取檔案名稱 (不含副檔名)
        """
        # 組合參數字串
        key_string = f"{symbol}_{period}_{interval}"
        # 產生 MD5 雜湊值作為檔名
        hash_value = hashlib.md5(key_string.encode()).hexdigest()[:16]
        return f"{symbol.replace('.', '_')}_{hash_value}"

    def _get_cache_paths(self, cache_key: str) -> Tuple[Path, Path]:
        """
        取得快取檔案路徑

        Returns:
            (數據檔案路徑, 元資料檔案路徑)
        """
        data_path = self.cache_dir / f"{cache_key}.parquet"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"
        return data_path, meta_path

    def is_cache_valid(self, symbol: str, period: str, interval: str) -> bool:
        """
        檢查快取是否有效 (12 小時內)

        快取判定邏輯：
        1. 檢查快取檔案是否存在
        2. 讀取元資料中的時間戳記
        3. 計算距離現在的時間差
        4. 若時間差 < 12 小時，快取有效

        Args:
            symbol: 股票代號
            period: 資料期間
            interval: K線週期

        Returns:
            True 表示快取有效，False 表示需要重新下載
        """
        cache_key = self._get_cache_key(symbol, period, interval)
        data_path, meta_path = self._get_cache_paths(cache_key)

        # 檢查檔案是否存在
        if not data_path.exists() or not meta_path.exists():
            return False

        try:
            # 讀取元資料
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # 解析快取時間戳記
            cache_time = datetime.fromisoformat(meta['timestamp'])
            current_time = datetime.now()

            # 計算時間差 (秒)
            time_diff = (current_time - cache_time).total_seconds()

            # 判斷是否在 12 小時內
            # 公式: time_diff < CACHE_EXPIRY_SECONDS (43200 秒 = 12 小時)
            is_valid = time_diff < CACHE_EXPIRY_SECONDS

            if is_valid:
                remaining_hours = (CACHE_EXPIRY_SECONDS - time_diff) / 3600
                print(f"[快取] 使用快取數據 (剩餘有效時間: {remaining_hours:.1f} 小時)")
            else:
                print(f"[快取] 快取已過期 (已超過 {CACHE_EXPIRY_HOURS} 小時)")

            return is_valid

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 元資料損壞，視為無效快取
            print(f"[快取] 元資料讀取失敗: {e}")
            return False

    def load_cache(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        從快取載入數據

        Args:
            symbol: 股票代號
            period: 資料期間
            interval: K線週期

        Returns:
            DataFrame 或 None (若快取無效)
        """
        if not self.is_cache_valid(symbol, period, interval):
            return None

        cache_key = self._get_cache_key(symbol, period, interval)
        data_path, _ = self._get_cache_paths(cache_key)

        try:
            df = pd.read_parquet(data_path)
            print(f"[快取] 成功載入 {len(df)} 根K線")
            return df
        except Exception as e:
            print(f"[快取] 載入失敗: {e}")
            return None

    def save_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        period: str,
        interval: str
    ) -> bool:
        """
        儲存數據到快取

        Args:
            df: 股票數據 DataFrame
            symbol: 股票代號
            period: 資料期間
            interval: K線週期

        Returns:
            True 表示儲存成功
        """
        cache_key = self._get_cache_key(symbol, period, interval)
        data_path, meta_path = self._get_cache_paths(cache_key)

        try:
            # 儲存數據 (使用 parquet 格式，高效且保留資料型態)
            df.to_parquet(data_path)

            # 儲存元資料
            meta = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'rows': len(df),
                'columns': list(df.columns)
            }

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"[快取] 已儲存至快取 (有效期 {CACHE_EXPIRY_HOURS} 小時)")
            return True

        except Exception as e:
            print(f"[快取] 儲存失敗: {e}")
            return False

    def clear_expired(self) -> int:
        """
        清除所有過期的快取檔案

        Returns:
            清除的檔案數量
        """
        cleared = 0
        current_time = datetime.now()

        for meta_file in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                cache_time = datetime.fromisoformat(meta['timestamp'])
                time_diff = (current_time - cache_time).total_seconds()

                if time_diff >= CACHE_EXPIRY_SECONDS:
                    # 刪除過期的數據檔和元資料檔
                    data_file = meta_file.with_suffix('').with_suffix('.parquet')
                    if data_file.exists():
                        data_file.unlink()
                    meta_file.unlink()
                    cleared += 1

            except Exception:
                continue

        if cleared > 0:
            print(f"[快取] 已清除 {cleared} 個過期快取")

        return cleared


# 全域快取管理器實例
_cache = DataCache()


# ============================================================
# 資料下載模組 (Data Download Module)
# ============================================================

def download_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    使用 yfinance 下載股票數據 (含 12 小時快取機制)

    Args:
        symbol: 股票代號 (如 "2330.TW")
        period: 資料期間 (如 "1y", "6mo", "3mo")
        interval: K線週期 (如 "1d", "1wk")
        use_cache: 是否使用快取 (預設 True)

    Returns:
        包含 OHLCV 的 DataFrame

    Raises:
        ValueError: 當無法取得數據時
    """
    try:
        # ========================================
        # 步驟 1: 嘗試從快取載入
        # ========================================
        if use_cache:
            cached_df = _cache.load_cache(symbol, period, interval)
            if cached_df is not None:
                return cached_df

        # ========================================
        # 步驟 2: 從 API 下載新數據
        # ========================================
        print(f"[下載] 正在從 Yahoo Finance 下載 {symbol} 數據...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"無法取得 {symbol} 的數據，請確認股票代號是否正確")

        # 確保欄位名稱統一 (小寫)
        df.columns = [col.lower() for col in df.columns]

        # 移除時區資訊以避免繪圖問題
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        print(f"[下載] 成功下載 {symbol} 數據: {len(df)} 根K線")

        # ========================================
        # 步驟 3: 儲存到快取
        # ========================================
        if use_cache:
            _cache.save_cache(df, symbol, period, interval)

        return df

    except ValueError:
        # 重新拋出自定義的錯誤
        raise
    except Exception as e:
        # 捕獲網路錯誤、API 錯誤等
        error_msg = f"下載數據時發生錯誤: {type(e).__name__}: {e}"
        print(f"[錯誤] {error_msg}")
        raise RuntimeError(error_msg) from e


# ============================================================
# 形態辨識演算法 (Pattern Recognition Algorithm)
# ============================================================

class PatternRecognizer:
    """技術形態辨識器"""

    def __init__(self, df: pd.DataFrame, pivot_window: int = 5, tolerance: float = 0.02):
        """
        初始化辨識器

        Args:
            df: OHLCV DataFrame
            pivot_window: 尋找轉折點的視窗大小
            tolerance: 價格比對容許誤差 (預設 2%)
        """
        self.df = df.copy()
        self.pivot_window = pivot_window
        self.tolerance = tolerance
        self.patterns: List[Pattern] = []

        # 尋找所有轉折點
        self.highs, self.lows = self._find_pivots()

    def _find_pivots(self) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        尋找價格轉折點 (局部極值)

        【數學原理】
        使用 scipy.signal.argrelextrema 尋找局部最大/最小值。
        對於索引 i，若滿足以下條件則為局部最大值：
            prices[i] > prices[i-w] 且 prices[i] > prices[i+w]
            其中 w = pivot_window (視窗大小)

        視窗大小決定了形態的「粒度」：
        - 較小的視窗 (如 3): 找到更多但較不顯著的轉折點
        - 較大的視窗 (如 10): 只找到最顯著的轉折點

        Returns:
            (高點列表, 低點列表), 每個點為 (索引, 價格) 的 tuple
        """
        prices_high = self.df['high'].values
        prices_low = self.df['low'].values

        # ========================================
        # 使用 argrelextrema 尋找局部極值
        # ========================================
        # np.greater: 比較函數，用於找局部最大值
        # np.less: 比較函數，用於找局部最小值
        # order: 比較的鄰居數量 (左右各 order 個點)
        #
        # 數學定義:
        # 局部最大值: P[i] > P[i-order], P[i-order+1], ..., P[i+order]
        # 局部最小值: P[i] < P[i-order], P[i-order+1], ..., P[i+order]
        high_indices = argrelextrema(prices_high, np.greater, order=self.pivot_window)[0]
        low_indices = argrelextrema(prices_low, np.less, order=self.pivot_window)[0]

        # 將索引和價格組合成 tuple 列表
        highs = [(int(i), float(prices_high[i])) for i in high_indices]
        lows = [(int(i), float(prices_low[i])) for i in low_indices]

        return highs, lows

    def _is_close(self, price1: float, price2: float, tol: Optional[float] = None) -> bool:
        """
        判斷兩個價格是否接近 (在容許誤差範圍內)

        【數學原理】
        使用「相對誤差」而非「絕對誤差」來比較價格。

        相對誤差公式:
            relative_error = |P1 - P2| / max(P1, P2)

        這種方法的優點是對不同價位的股票都適用：
        - 股價 1000 元，2% 容許誤差 = 20 元
        - 股價 50 元，2% 容許誤差 = 1 元

        判定條件: relative_error <= tolerance

        Args:
            price1: 第一個價格 (P1)
            price2: 第二個價格 (P2)
            tol: 容許誤差百分比 (預設使用 self.tolerance = 0.02 即 2%)

        Returns:
            True 表示兩價格在容許範圍內，視為「接近」
        """
        if tol is None:
            tol = self.tolerance

        # 取較大值作為分母，避免除以零
        max_price = max(price1, price2)
        if max_price == 0:
            return True

        # 計算相對誤差並與容許值比較
        # 公式: |P1 - P2| / max(P1, P2) <= tol
        relative_error = abs(price1 - price2) / max_price
        return relative_error <= tol

    def find_head_and_shoulders(self) -> List[Pattern]:
        """
        辨識頭肩頂形態 (Head and Shoulders Top)

        條件:
        1. 三個高點: 左肩 < 頭部 > 右肩
        2. 左肩和右肩價格接近
        3. 兩個低點形成頸線
        """
        patterns = []

        if len(self.highs) < 3:
            return patterns

        for i in range(len(self.highs) - 2):
            left_shoulder = self.highs[i]
            head = self.highs[i + 1]
            right_shoulder = self.highs[i + 2]

            # 條件1: 頭部必須高於兩肩
            if not (head[1] > left_shoulder[1] and head[1] > right_shoulder[1]):
                continue

            # 條件2: 兩肩價格接近 (使用較寬鬆的容許度)
            if not self._is_close(left_shoulder[1], right_shoulder[1], tol=0.05):
                continue

            # 尋找頸線 (兩肩之間的低點)
            neckline_lows = [
                low for low in self.lows
                if left_shoulder[0] < low[0] < right_shoulder[0]
            ]

            if len(neckline_lows) < 2:
                continue

            # 計算信心度 (頭部相對於肩部的突出程度)
            shoulder_avg = (left_shoulder[1] + right_shoulder[1]) / 2
            prominence = (head[1] - shoulder_avg) / shoulder_avg
            confidence = min(prominence * 5, 1.0)  # 正規化到 0-1

            pattern = Pattern(
                name="頭肩頂",
                start_idx=left_shoulder[0],
                end_idx=right_shoulder[0],
                key_points=[left_shoulder, head, right_shoulder],
                pattern_type="bearish",
                confidence=confidence
            )
            patterns.append(pattern)

        return patterns

    def find_inverse_head_and_shoulders(self) -> List[Pattern]:
        """
        辨識頭肩底形態 (Inverse Head and Shoulders)

        條件:
        1. 三個低點: 左肩 > 頭部 < 右肩
        2. 左肩和右肩價格接近
        """
        patterns = []

        if len(self.lows) < 3:
            return patterns

        for i in range(len(self.lows) - 2):
            left_shoulder = self.lows[i]
            head = self.lows[i + 1]
            right_shoulder = self.lows[i + 2]

            # 條件1: 頭部必須低於兩肩
            if not (head[1] < left_shoulder[1] and head[1] < right_shoulder[1]):
                continue

            # 條件2: 兩肩價格接近
            if not self._is_close(left_shoulder[1], right_shoulder[1], tol=0.05):
                continue

            # 計算信心度
            shoulder_avg = (left_shoulder[1] + right_shoulder[1]) / 2
            prominence = (shoulder_avg - head[1]) / shoulder_avg
            confidence = min(prominence * 5, 1.0)

            pattern = Pattern(
                name="頭肩底",
                start_idx=left_shoulder[0],
                end_idx=right_shoulder[0],
                key_points=[left_shoulder, head, right_shoulder],
                pattern_type="bullish",
                confidence=confidence
            )
            patterns.append(pattern)

        return patterns

    def find_double_top(self) -> List[Pattern]:
        """
        辨識雙頂形態 (Double Top / M頭)

        條件:
        1. 兩個接近的高點
        2. 中間有明顯的回調低點
        """
        patterns = []

        if len(self.highs) < 2:
            return patterns

        for i in range(len(self.highs) - 1):
            top1 = self.highs[i]
            top2 = self.highs[i + 1]

            # 條件1: 兩個高點價格接近
            if not self._is_close(top1[1], top2[1], tol=0.03):
                continue

            # 條件2: 兩高點之間要有足夠的距離 (至少10根K線)
            if top2[0] - top1[0] < 10:
                continue

            # 尋找中間的低點
            middle_lows = [
                low for low in self.lows
                if top1[0] < low[0] < top2[0]
            ]

            if not middle_lows:
                continue

            # 取最低的那個點
            valley = min(middle_lows, key=lambda x: x[1])

            # 計算回調深度作為信心度依據
            top_avg = (top1[1] + top2[1]) / 2
            retracement = (top_avg - valley[1]) / top_avg

            # 回調至少要有3%才算有效
            if retracement < 0.03:
                continue

            confidence = min(retracement * 3, 1.0)

            pattern = Pattern(
                name="雙頂(M頭)",
                start_idx=top1[0],
                end_idx=top2[0],
                key_points=[top1, valley, top2],
                pattern_type="bearish",
                confidence=confidence
            )
            patterns.append(pattern)

        return patterns

    def find_double_bottom(self) -> List[Pattern]:
        """
        辨識雙底形態 (Double Bottom / W底)

        條件:
        1. 兩個接近的低點
        2. 中間有明顯的反彈高點
        """
        patterns = []

        if len(self.lows) < 2:
            return patterns

        for i in range(len(self.lows) - 1):
            bottom1 = self.lows[i]
            bottom2 = self.lows[i + 1]

            # 條件1: 兩個低點價格接近
            if not self._is_close(bottom1[1], bottom2[1], tol=0.03):
                continue

            # 條件2: 兩低點之間要有足夠的距離
            if bottom2[0] - bottom1[0] < 10:
                continue

            # 尋找中間的高點
            middle_highs = [
                high for high in self.highs
                if bottom1[0] < high[0] < bottom2[0]
            ]

            if not middle_highs:
                continue

            # 取最高的那個點
            peak = max(middle_highs, key=lambda x: x[1])

            # 計算反彈幅度
            bottom_avg = (bottom1[1] + bottom2[1]) / 2
            bounce = (peak[1] - bottom_avg) / bottom_avg

            if bounce < 0.03:
                continue

            confidence = min(bounce * 3, 1.0)

            pattern = Pattern(
                name="雙底(W底)",
                start_idx=bottom1[0],
                end_idx=bottom2[0],
                key_points=[bottom1, peak, bottom2],
                pattern_type="bullish",
                confidence=confidence
            )
            patterns.append(pattern)

        return patterns

    def find_triangle(self) -> List[Pattern]:
        """
        辨識三角收斂形態 (Triangle / Consolidation)

        使用線性回歸判斷高點下降趨勢和低點上升趨勢
        當兩條趨勢線收斂時，形成三角形
        """
        patterns = []

        if len(self.highs) < 3 or len(self.lows) < 3:
            return patterns

        # 需要至少有連續的轉折點來形成三角形
        # 使用滑動視窗尋找收斂區間
        window_size = min(len(self.highs), len(self.lows), 5)

        for start in range(len(self.highs) - window_size + 1):
            # 取得視窗內的高點和低點
            window_highs = self.highs[start:start + window_size]

            # 找出對應時間範圍內的低點
            start_idx = window_highs[0][0]
            end_idx = window_highs[-1][0]

            window_lows = [
                low for low in self.lows
                if start_idx <= low[0] <= end_idx
            ]

            if len(window_lows) < 3:
                continue

            # 計算高點的斜率 (應該是負的或接近零)
            high_indices = np.array([h[0] for h in window_highs])
            high_prices = np.array([h[1] for h in window_highs])

            if len(high_indices) > 1:
                high_slope = np.polyfit(high_indices, high_prices, 1)[0]
            else:
                continue

            # 計算低點的斜率 (應該是正的或接近零)
            low_indices = np.array([l[0] for l in window_lows])
            low_prices = np.array([l[1] for l in window_lows])

            if len(low_indices) > 1:
                low_slope = np.polyfit(low_indices, low_prices, 1)[0]
            else:
                continue

            # 判斷是否收斂 (高點下降 + 低點上升，或其中一條水平)
            is_converging = (high_slope <= 0 and low_slope >= 0)

            # 判斷三角形類型
            if is_converging:
                if high_slope < -0.001 and low_slope > 0.001:
                    triangle_type = "對稱三角"
                elif abs(high_slope) < 0.001 and low_slope > 0.001:
                    triangle_type = "上升三角"
                elif high_slope < -0.001 and abs(low_slope) < 0.001:
                    triangle_type = "下降三角"
                else:
                    continue

                # 計算收斂程度作為信心度
                price_range_start = window_highs[0][1] - window_lows[0][1]
                price_range_end = window_highs[-1][1] - window_lows[-1][1]

                if price_range_start > 0:
                    convergence = 1 - (price_range_end / price_range_start)
                    confidence = min(max(convergence, 0), 1.0)
                else:
                    confidence = 0.5

                # 決定看多或看空
                if triangle_type == "上升三角":
                    pattern_type = "bullish"
                elif triangle_type == "下降三角":
                    pattern_type = "bearish"
                else:
                    pattern_type = "neutral"

                pattern = Pattern(
                    name=triangle_type,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points=window_highs + window_lows,
                    pattern_type=pattern_type,
                    confidence=confidence
                )
                patterns.append(pattern)

        return patterns

    def find_channel(self) -> List[Pattern]:
        """
        辨識通道形態 (Channel)

        【形態說明】
        - 上升通道: 高點和低點都呈現上升趨勢，價格在兩條平行線之間波動
        - 下降通道: 高點和低點都呈現下降趨勢
        - 橫向通道: 價格在水平區間內震盪

        【數學原理】
        使用線性迴歸 (Linear Regression) 擬合高點和低點的趨勢線：
            y = mx + b
            其中 m = 斜率 (slope), b = 截距 (intercept)

        通道判定條件:
        1. 高點趨勢線和低點趨勢線的斜率相近 (平行)
        2. 使用 R² (決定係數) 評估擬合程度
        """
        patterns = []

        # 至少需要 3 個高點和 3 個低點才能可靠地擬合趨勢線
        if len(self.highs) < 3 or len(self.lows) < 3:
            return patterns

        # ========================================
        # 步驟 1: 準備數據點
        # ========================================
        high_indices = np.array([h[0] for h in self.highs])  # x 座標 (時間索引)
        high_prices = np.array([h[1] for h in self.highs])   # y 座標 (價格)
        low_indices = np.array([l[0] for l in self.lows])
        low_prices = np.array([l[1] for l in self.lows])

        # ========================================
        # 步驟 2: 使用最小二乘法擬合趨勢線
        # ========================================
        # np.polyfit(x, y, 1) 回傳一次多項式係數 [斜率, 截距]
        # 數學原理: 最小化 Σ(y_i - (m*x_i + b))² 的誤差平方和
        high_slope, high_intercept = np.polyfit(high_indices, high_prices, 1)
        low_slope, low_intercept = np.polyfit(low_indices, low_prices, 1)

        # ========================================
        # 步驟 3: 檢查兩線是否平行
        # ========================================
        # 平行判定: 斜率差異 / 平均斜率 <= 0.5 (50% 容許度)
        slope_diff = abs(high_slope - low_slope)
        avg_slope = (abs(high_slope) + abs(low_slope)) / 2

        if avg_slope > 0 and slope_diff / avg_slope > 0.5:
            # 斜率差異太大，不構成平行通道
            return patterns

        # ========================================
        # 步驟 4: 判斷通道類型
        # ========================================
        avg_slope = (high_slope + low_slope) / 2

        # 斜率閾值 0.001 用於區分趨勢方向
        # 正斜率 > 0.001: 上升趨勢
        # 負斜率 < -0.001: 下降趨勢
        # 其他: 橫向震盪
        if avg_slope > 0.001:
            channel_type = "上升通道"
            pattern_type = "bullish"
        elif avg_slope < -0.001:
            channel_type = "下降通道"
            pattern_type = "bearish"
        else:
            channel_type = "橫向通道"
            pattern_type = "neutral"

        # ========================================
        # 步驟 5: 計算 R² (決定係數) 作為信心度
        # ========================================
        # linregress 回傳: slope, intercept, r_value, p_value, std_err
        # R² = r_value² 表示模型解釋變異的比例
        # R² 越接近 1，表示擬合越好

        _, _, r_high, _, _ = linregress(high_indices, high_prices)
        _, _, r_low, _, _ = linregress(low_indices, low_prices)

        # 取兩條線 R 值的平均作為整體信心度
        # 使用 |r| 而非 r² 是為了保持敏感度
        confidence = (abs(r_high) + abs(r_low)) / 2

        # 只有當擬合度 > 0.5 時才報告 (避免噪音)
        if confidence > 0.5:
            start_idx = min(self.highs[0][0], self.lows[0][0])
            end_idx = max(self.highs[-1][0], self.lows[-1][0])

            pattern = Pattern(
                name=channel_type,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points=self.highs + self.lows,
                pattern_type=pattern_type,
                confidence=confidence
            )
            patterns.append(pattern)

        return patterns

    def analyze(self) -> List[Pattern]:
        """
        執行完整的形態分析

        Returns:
            所有辨識到的形態列表
        """
        self.patterns = []

        # 執行所有形態辨識
        self.patterns.extend(self.find_head_and_shoulders())
        self.patterns.extend(self.find_inverse_head_and_shoulders())
        self.patterns.extend(self.find_double_top())
        self.patterns.extend(self.find_double_bottom())
        self.patterns.extend(self.find_triangle())
        self.patterns.extend(self.find_channel())

        # 按信心度排序
        self.patterns.sort(key=lambda x: x.confidence, reverse=True)

        return self.patterns

    def get_summary(self) -> str:
        """產生分析摘要"""
        if not self.patterns:
            return "未辨識到任何技術形態"

        lines = ["=== 技術形態分析結果 ===\n"]

        for i, p in enumerate(self.patterns, 1):
            type_str = "看多" if p.pattern_type == "bullish" else "看空" if p.pattern_type == "bearish" else "中性"
            lines.append(
                f"{i}. {p.name} ({type_str})\n"
                f"   區間: 第 {p.start_idx} 至 {p.end_idx} 根K線\n"
                f"   信心度: {p.confidence:.1%}\n"
            )

        return "\n".join(lines)


# ============================================================
# 繪圖模組 (Plotting Module)
# ============================================================

def plot_patterns(
    df: pd.DataFrame,
    patterns: List[Pattern],
    symbol: str = "Stock",
    recognizer: Optional[PatternRecognizer] = None,
    save_path: Optional[str] = None
) -> None:
    """
    繪製K線圖並標註形態

    Args:
        df: OHLCV DataFrame
        patterns: 辨識到的形態列表
        symbol: 股票代號 (用於標題)
        recognizer: PatternRecognizer 實例 (用於取得轉折點)
        save_path: 儲存路徑 (若指定則儲存圖片而非顯示)
    """
    try:
        # 準備 mplfinance 需要的欄位名稱
        plot_df = df.copy()
        plot_df.columns = [col.capitalize() for col in plot_df.columns]

        # 建立標註點
        apds = []  # additional plots

        # 標註轉折點
        if recognizer is not None:
            # 高點標註
            high_markers = np.full(len(df), np.nan)
            for idx, price in recognizer.highs:
                if 0 <= idx < len(df):
                    high_markers[idx] = price * 1.01  # 稍微上移

            apds.append(mpf.make_addplot(
                high_markers,
                type='scatter',
                markersize=50,
                marker='v',
                color='red',
                alpha=0.7
            ))

            # 低點標註
            low_markers = np.full(len(df), np.nan)
            for idx, price in recognizer.lows:
                if 0 <= idx < len(df):
                    low_markers[idx] = price * 0.99  # 稍微下移

            apds.append(mpf.make_addplot(
                low_markers,
                type='scatter',
                markersize=50,
                marker='^',
                color='green',
                alpha=0.7
            ))

        # 設定圖表樣式
        mc = mpf.make_marketcolors(
            up='red',
            down='green',
            edge='inherit',
            wick='inherit',
            volume='in'
        )

        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=':',
            gridcolor='gray',
            figcolor='white',
            facecolor='white',
            rc={'font.family': 'Microsoft JhengHei'}
        )

        # 繪製主圖
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            style=style,
            title=f'\n{symbol} 技術形態分析',
            ylabel='價格',
            ylabel_lower='成交量',
            volume=True,
            addplot=apds if apds else None,
            figsize=(14, 8),
            returnfig=True
        )

        # 在圖上標註形態區域
        ax = axes[0]

        colors = {
            'bullish': 'green',
            'bearish': 'red',
            'neutral': 'blue'
        }

        for pattern in patterns[:5]:  # 只顯示前5個最有信心的形態
            color = colors.get(pattern.pattern_type, 'gray')

            # 標註形態區間
            ax.axvspan(
                pattern.start_idx,
                pattern.end_idx,
                alpha=0.15,
                color=color,
                label=f"{pattern.name} ({pattern.confidence:.0%})"
            )

            # 連接關鍵點
            if len(pattern.key_points) >= 2:
                x_coords = [p[0] for p in pattern.key_points[:3]]
                y_coords = [p[1] for p in pattern.key_points[:3]]
                ax.plot(x_coords, y_coords, 'o-', color=color, linewidth=2, markersize=8)

        # 添加圖例
        if patterns:
            ax.legend(loc='upper left', fontsize=9)

        # 添加形態摘要文字框
        if patterns:
            summary_lines = []
            for p in patterns[:3]:
                type_emoji = "▲" if p.pattern_type == "bullish" else "▼" if p.pattern_type == "bearish" else "◆"
                summary_lines.append(f"{type_emoji} {p.name}: {p.confidence:.0%}")

            summary_text = "\n".join(summary_lines)

            ax.text(
                0.02, 0.98,
                summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='Microsoft JhengHei'  # 支援中文
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"繪圖錯誤: {e}")
        # 備用方案：使用基本 matplotlib
        _plot_fallback(df, patterns, symbol, save_path)


def _plot_fallback(
    df: pd.DataFrame,
    patterns: List[Pattern],
    symbol: str,
    save_path: Optional[str] = None
) -> None:
    """備用繪圖方案 (純 matplotlib)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # 繪製收盤價
    ax1.plot(df.index, df['close'], 'b-', linewidth=1, label='收盤價')
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.3)

    # 標註形態
    colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}

    for pattern in patterns[:5]:
        start_date = df.index[pattern.start_idx]
        end_date = df.index[min(pattern.end_idx, len(df)-1)]
        color = colors.get(pattern.pattern_type, 'gray')

        ax1.axvspan(start_date, end_date, alpha=0.2, color=color,
                   label=f"{pattern.name} ({pattern.confidence:.0%})")

    ax1.set_title(f'{symbol} 技術形態分析')
    ax1.set_ylabel('價格')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 繪製成交量
    ax2.bar(df.index, df['volume'], color='gray', alpha=0.5)
    ax2.set_ylabel('成交量')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"圖表已儲存至: {save_path}")
        plt.close()
    else:
        plt.show()


# ============================================================
# 主程式 (Main)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="股票技術形態辨識系統")
    parser.add_argument("--symbol", "-s", default="2330.TW", help="股票代號 (預設: 2330.TW)")
    parser.add_argument("--period", "-p", default="1y", help="資料期間 (預設: 1y)")
    parser.add_argument("--save", "-o", default=None, help="儲存圖片路徑 (若不指定則顯示視窗)")
    parser.add_argument("--window", "-w", type=int, default=5, help="轉折點視窗大小 (預設: 5)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.02, help="價格容許誤差 (預設: 0.02)")

    args = parser.parse_args()

    try:
        print(f"開始分析 {args.symbol}...")
        print("-" * 40)

        # 1. 下載數據
        df = download_stock_data(args.symbol, period=args.period)

        # 2. 執行形態辨識
        recognizer = PatternRecognizer(
            df,
            pivot_window=args.window,
            tolerance=args.tolerance
        )

        patterns = recognizer.analyze()

        # 3. 輸出分析結果
        print(recognizer.get_summary())

        # 4. 繪製圖表
        plot_patterns(df, patterns, args.symbol, recognizer, save_path=args.save)

    except Exception as e:
        print(f"程式執行錯誤: {e}")
        import traceback
        traceback.print_exc()
