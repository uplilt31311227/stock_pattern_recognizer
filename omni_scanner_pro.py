# -*- coding: utf-8 -*-
"""
================================================================================
Omni-Scanner Pro - 高階市場形態掃描系統
================================================================================
專為 Google Colab 設計的一鍵掃描工具
支援台股 (FinMind) 與美股 (Tiingo) 數據源
包含 12 小時智慧快取、形態評分引擎、互動式儀表板

Author: Quant Architect
Version: 1.0.0
================================================================================
"""

# ============================================================
# [區塊 0] 環境安裝 (僅在 Colab 執行)
# ============================================================
"""
# 在 Colab 中執行以下指令安裝套件：
!pip install FinMind pandas-datareader plotly pandas_ta tqdm ipywidgets -q
"""

# ============================================================
# [區塊 1] 套件匯入 (Library Imports)
# ============================================================

# 標準函式庫
import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# 數據處理
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# 技術分析 (選用)
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("[警告] pandas_ta 未安裝，將使用內建 ATR 計算")

# 數據源
try:
    from FinMind.data import DataLoader as FinMindLoader
    HAS_FINMIND = True
except ImportError:
    HAS_FINMIND = False
    print("[警告] FinMind 未安裝，台股功能將無法使用")

# 美股數據源: 優先使用 yfinance (更穩定)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("[警告] yfinance 未安裝，美股功能將無法使用")

# 備用: pandas-datareader
HAS_PDR = False  # 暫時停用，因為與新版 pandas 不相容
# try:
#     import pandas_datareader as pdr
#     HAS_PDR = True
# except ImportError:
#     HAS_PDR = False

# 視覺化
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 互動元件
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("[警告] ipywidgets 未安裝，將使用命令列模式")

# 進度條
from tqdm.auto import tqdm

# 忽略警告
warnings.filterwarnings('ignore')


# ============================================================
# [區塊 2] 全域設定 (Global Configuration)
# ============================================================

# 快取設定
CACHE_DIR = Path("./cache")
CACHE_EXPIRY_HOURS = 12  # 快取有效期：12 小時

# 評分門檻
SCORE_THRESHOLD = 75  # 只顯示總分 > 75 的形態

# 評分權重配置
SCORING_WEIGHTS = {
    'geometry': 0.50,      # 幾何標準度 50%
    'volume': 0.20,        # 帶量突破 20%
    'order_block': 0.15,   # 訂單塊共振 15%
    'fibonacci': 0.15      # 斐波納契 15%
}

# 熱門股清單
WATCHLIST = {
    'TW': ['2330', '2454', '2603', '2317', '2881', '2303', '2882', '1301'],
    'US': ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'META']
}

# 如需讀取外部清單，可使用：
# WATCHLIST['TW'] = pd.read_csv('tw_tickers.csv')['ticker'].tolist()
# WATCHLIST['US'] = pd.read_csv('us_tickers.csv')['ticker'].tolist()


# ============================================================
# [區塊 3] 資料結構定義 (Data Structures)
# ============================================================

@dataclass
class PatternResult:
    """形態辨識結果資料結構"""
    ticker: str                    # 股票代號
    market: str                    # 市場 (TW/US)
    pattern_name: str              # 形態名稱
    direction: str                 # 方向 (bullish/bearish)
    score: float                   # 總分
    current_price: float           # 現價
    signal_date: datetime          # 訊號日期
    key_levels: Dict[str, float]   # 關鍵價位
    score_breakdown: Dict[str, float] = field(default_factory=dict)  # 分項得分


# ============================================================
# [Task 1] 智慧數據管理器 (DataManager Class)
# ============================================================

class DataManager:
    """
    智慧數據管理器

    功能：
    1. 統一管理台股 (FinMind) 和美股 (Tiingo) 數據源
    2. 實作 12 小時檔案快取機制，減少 API 呼叫
    3. 標準化輸出格式 (OHLCV + Datetime Index)
    """

    def __init__(self, finmind_token: str = "", tiingo_key: str = ""):
        """
        初始化數據管理器

        Args:
            finmind_token: FinMind API Token (台股用)
            tiingo_key: Tiingo API Key (美股用)
        """
        self.finmind_token = finmind_token
        self.tiingo_key = tiingo_key

        # 建立快取目錄
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 初始化 FinMind Loader
        if HAS_FINMIND and finmind_token:
            self.finmind = FinMindLoader()
            self.finmind.login_by_token(api_token=finmind_token)
        else:
            self.finmind = None

    def _get_cache_path(self, ticker: str, market: str) -> Path:
        """
        產生快取檔案路徑

        檔名格式: {market}_{ticker}.csv
        例如: TW_2330.csv, US_AAPL.csv
        """
        safe_ticker = ticker.replace('.', '_').replace('/', '_')
        return CACHE_DIR / f"{market}_{safe_ticker}.csv"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        檢查快取是否有效 (12 小時內)

        【快取判定邏輯】
        1. 檢查檔案是否存在
        2. 取得檔案最後修改時間 (mtime)
        3. 計算時間差: Δt = 現在時間 - 修改時間
        4. 若 Δt < 12 小時，快取有效

        數學公式:
            is_valid = (datetime.now() - file_mtime).total_seconds() < 12 * 3600
        """
        if not cache_path.exists():
            return False

        # 取得檔案修改時間
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        current_time = datetime.now()

        # 計算時間差 (秒)
        time_diff_seconds = (current_time - file_mtime).total_seconds()

        # 判斷是否在 12 小時內 (12 * 60 * 60 = 43200 秒)
        cache_expiry_seconds = CACHE_EXPIRY_HOURS * 3600
        is_valid = time_diff_seconds < cache_expiry_seconds

        if is_valid:
            remaining_hours = (cache_expiry_seconds - time_diff_seconds) / 3600
            print(f"  [快取] 使用快取 (剩餘 {remaining_hours:.1f} 小時)")

        return is_valid

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        標準化 DataFrame 欄位名稱

        統一輸出格式: ['Open', 'High', 'Low', 'Close', 'Volume']
        索引: DatetimeIndex
        """
        # 欄位名稱映射表
        column_mappings = {
            # FinMind 格式
            'open': 'Open', 'max': 'High', 'min': 'Low',
            'close': 'Close', 'Trading_Volume': 'Volume',
            # Tiingo 格式
            'adjOpen': 'Open', 'adjHigh': 'High',
            'adjLow': 'Low', 'adjClose': 'Close', 'adjVolume': 'Volume',
            # 通用格式 (小寫)
            'high': 'High', 'low': 'Low', 'volume': 'Volume'
        }

        # 重命名欄位
        df = df.rename(columns=column_mappings)

        # 確保必要欄位存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                # 嘗試不區分大小寫的匹配
                for orig_col in df.columns:
                    if orig_col.lower() == col.lower():
                        df = df.rename(columns={orig_col: col})
                        break

        # 只保留必要欄位
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols].copy()

        # 確保索引為 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')

        # 移除時區資訊
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 排序 (由舊到新)
        df = df.sort_index()

        return df

    def _fetch_tw_stock(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        從 FinMind 下載台股數據

        Args:
            ticker: 股票代號 (如 '2330')
            days: 取得天數 (預設 365 天)
        """
        if not HAS_FINMIND:
            raise RuntimeError("FinMind 未安裝，請執行: pip install FinMind")

        if not self.finmind:
            raise RuntimeError("請先設定 FinMind Token")

        # 計算日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 呼叫 FinMind API
        df = self.finmind.taiwan_stock_daily(
            stock_id=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df.empty:
            raise ValueError(f"無法取得 {ticker} 的數據")

        # 設定日期索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df

    def _fetch_us_stock(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        下載美股數據

        優先使用 yfinance (免費且穩定)
        若有設定 Tiingo Key 則可作為備用

        Args:
            ticker: 股票代號 (如 'AAPL')
            days: 取得天數 (預設 365 天)
        """
        # 優先使用 yfinance
        if HAS_YFINANCE:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=f"{days}d")

                if df.empty:
                    raise ValueError(f"yfinance 無法取得 {ticker}")

                # yfinance 欄位名稱已經標準化
                return df

            except Exception as e:
                print(f"  [yfinance] {ticker} 取得失敗: {e}")

        # 備用: Tiingo (需要 API Key)
        if HAS_PDR and self.tiingo_key:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                df = pdr.get_data_tiingo(
                    ticker, start=start_date, end=end_date,
                    api_key=self.tiingo_key
                )

                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)

                return df

            except Exception as e:
                print(f"  [Tiingo] {ticker} 取得失敗: {e}")

        raise RuntimeError(f"無法取得 {ticker} 數據，請確認 yfinance 已安裝")

    def get_stock_data(
        self,
        ticker: str,
        market: str = 'TW',
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        取得股票數據 (主要方法)

        【執行流程】
        1. 檢查快取是否存在且有效 (12 小時內)
        2. 若有效 → 讀取 CSV 快取
        3. 若無效 → 呼叫 API 下載 → 儲存 CSV → 回傳

        Args:
            ticker: 股票代號
            market: 市場 ('TW' 或 'US')
            use_cache: 是否使用快取 (預設 True)

        Returns:
            標準化的 OHLCV DataFrame，失敗時回傳 None
        """
        cache_path = self._get_cache_path(ticker, market)

        try:
            # ========================================
            # 步驟 1: 檢查快取
            # ========================================
            if use_cache and self._is_cache_valid(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return self._standardize_columns(df)

            # ========================================
            # 步驟 2: 從 API 下載
            # ========================================
            print(f"  [下載] 正在從 API 取得 {ticker} 數據...")

            if market.upper() == 'TW':
                df = self._fetch_tw_stock(ticker)
            elif market.upper() == 'US':
                df = self._fetch_us_stock(ticker)
            else:
                raise ValueError(f"不支援的市場: {market}")

            # ========================================
            # 步驟 3: 標準化並儲存快取
            # ========================================
            df = self._standardize_columns(df)

            if use_cache:
                df.to_csv(cache_path)
                print(f"  [快取] 已儲存至 {cache_path}")

            return df

        except Exception as e:
            print(f"  [錯誤] {ticker}: {type(e).__name__}: {e}")
            return None


# ============================================================
# [Task 2] 形態識別與評分引擎 (PatternEngine Class)
# ============================================================

class PatternEngine:
    """
    形態識別與評分引擎

    功能：
    1. 識別技術形態 (W底、M頭、頭肩底/頂)
    2. 計算多維度評分 (幾何、量能、訂單塊、斐波納契)
    3. 篩選高品質形態 (Score >= 75)

    【評分權重配置】
    - 幾何標準度: 50% (形態轉折點的精確度)
    - 帶量突破: 20% (突破時的成交量放大)
    - 訂單塊共振: 15% (轉折點與訂單塊的重疊)
    - 斐波納契: 15% (轉折點與 Fibo 回撤位的吻合)
    """

    def __init__(self, pivot_window: int = 5, tolerance: float = 0.03):
        """
        初始化引擎

        Args:
            pivot_window: 轉折點偵測視窗 (預設 5)
            tolerance: 價格容許誤差 (預設 3%)
        """
        self.pivot_window = pivot_window
        self.tolerance = tolerance

    # ========================================
    # 技術指標計算
    # ========================================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        計算 ATR (Average True Range)

        【數學公式】
        True Range (TR) = max(
            High - Low,
            |High - Previous Close|,
            |Low - Previous Close|
        )
        ATR = SMA(TR, period)

        ATR 用於衡量價格波動性，判斷形態的有效性
        """
        if HAS_PANDAS_TA:
            return ta.atr(df['High'], df['Low'], df['Close'], length=period)

        # 內建計算
        high = df['High']
        low = df['Low']
        close = df['Close']

        # 計算 True Range 的三個組成部分
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        # True Range = 三者取最大值
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = TR 的簡單移動平均
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_volume_ma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """計算成交量移動平均"""
        return df['Volume'].rolling(window=period).mean()

    def _find_pivots(self, df: pd.DataFrame) -> Tuple[List, List]:
        """
        尋找價格轉折點 (Pivot High / Pivot Low)

        【演算法說明】
        使用 scipy.signal.argrelextrema 尋找局部極值

        對於索引 i，若滿足以下條件則為 Pivot High:
            prices[i] > prices[i-w], prices[i-w+1], ..., prices[i+w]
            其中 w = pivot_window

        Returns:
            (pivot_highs, pivot_lows): 兩個列表，每個元素為 (索引, 價格)
        """
        prices_high = df['High'].values
        prices_low = df['Low'].values

        # 使用 argrelextrema 尋找局部極值
        high_indices = argrelextrema(prices_high, np.greater, order=self.pivot_window)[0]
        low_indices = argrelextrema(prices_low, np.less, order=self.pivot_window)[0]

        pivot_highs = [(int(i), float(prices_high[i])) for i in high_indices]
        pivot_lows = [(int(i), float(prices_low[i])) for i in low_indices]

        return pivot_highs, pivot_lows

    def _find_order_blocks(self, df: pd.DataFrame, pivot_idx: int, lookback: int = 10) -> Optional[Dict]:
        """
        識別訂單塊 (Order Block)

        【訂單塊定義】
        在趨勢反轉前的最後一根反向 K 線區域
        - 在上漲前的訂單塊: 最後一根陰線的範圍
        - 在下跌前的訂單塊: 最後一根陽線的範圍

        這些區域通常是機構大單進場的位置，
        價格回測時常獲得支撐/壓力

        Args:
            df: OHLCV DataFrame
            pivot_idx: 轉折點索引
            lookback: 向前搜尋的 K 線數量

        Returns:
            {'high': 上緣, 'low': 下緣, 'idx': 索引} 或 None
        """
        if pivot_idx < lookback:
            return None

        # 取得轉折點前的 K 線
        start_idx = max(0, pivot_idx - lookback)
        subset = df.iloc[start_idx:pivot_idx]

        if len(subset) < 2:
            return None

        # 判斷轉折方向
        pivot_price = df.iloc[pivot_idx]['Close']
        pre_price = df.iloc[start_idx]['Close']

        if pivot_price > pre_price:
            # 上漲趨勢 → 找最後一根陰線 (看漲訂單塊)
            bearish_candles = subset[subset['Close'] < subset['Open']]
            if bearish_candles.empty:
                return None
            ob_idx = bearish_candles.index[-1]
            ob_row = df.loc[ob_idx]
        else:
            # 下跌趨勢 → 找最後一根陽線 (看跌訂單塊)
            bullish_candles = subset[subset['Close'] > subset['Open']]
            if bullish_candles.empty:
                return None
            ob_idx = bullish_candles.index[-1]
            ob_row = df.loc[ob_idx]

        return {
            'high': float(ob_row['High']),
            'low': float(ob_row['Low']),
            'idx': ob_idx
        }

    def _calculate_fibonacci(self, high: float, low: float) -> Dict[str, float]:
        """
        計算斐波納契回撤位

        【數學公式】
        Fibo Level = High - (High - Low) × Ratio

        常用回撤比例:
        - 0.236 (23.6%)
        - 0.382 (38.2%)
        - 0.500 (50.0%)
        - 0.618 (61.8%) - 黃金比例
        - 0.786 (78.6%)

        這些比例源自斐波納契數列的收斂特性
        """
        diff = high - low

        return {
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382,
            '0.500': high - diff * 0.500,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786
        }

    def _is_near_level(self, price: float, level: float, tolerance: float = 0.02) -> bool:
        """
        判斷價格是否接近某個關鍵價位

        使用相對誤差: |price - level| / level <= tolerance
        """
        if level == 0:
            return False
        return abs(price - level) / level <= tolerance

    # ========================================
    # 形態識別邏輯
    # ========================================

    def _detect_double_bottom(self, df: pd.DataFrame, lows: List) -> List[Dict]:
        """
        識別 W 底 (雙底) 形態

        【形態特徵】
        1. 兩個相近的低點 (L1, L2)
        2. 中間有一個反彈高點 (Neckline)
        3. L1 和 L2 價格差異在容許範圍內 (3%)

        【看漲訊號】
        價格突破頸線時確認形態完成
        """
        patterns = []

        if len(lows) < 2:
            return patterns

        for i in range(len(lows) - 1):
            l1_idx, l1_price = lows[i]
            l2_idx, l2_price = lows[i + 1]

            # 條件1: 兩低點距離足夠 (至少 10 根 K 線)
            if l2_idx - l1_idx < 10:
                continue

            # 條件2: 兩低點價格接近 (誤差 < 3%)
            price_diff = abs(l1_price - l2_price) / max(l1_price, l2_price)
            if price_diff > self.tolerance:
                continue

            # 尋找中間的高點 (頸線)
            middle_highs = df.iloc[l1_idx:l2_idx]['High']
            if len(middle_highs) < 3:
                continue

            neckline_idx = middle_highs.idxmax()
            neckline_price = middle_highs.max()

            # 計算形態深度
            avg_low = (l1_price + l2_price) / 2
            depth = (neckline_price - avg_low) / avg_low

            if depth < 0.03:  # 深度至少 3%
                continue

            patterns.append({
                'type': 'W底',
                'direction': 'bullish',
                'l1': (l1_idx, l1_price),
                'l2': (l2_idx, l2_price),
                'neckline': (neckline_idx, neckline_price),
                'depth': depth
            })

        return patterns

    def _detect_double_top(self, df: pd.DataFrame, highs: List) -> List[Dict]:
        """
        識別 M 頭 (雙頂) 形態

        【形態特徵】
        1. 兩個相近的高點 (H1, H2)
        2. 中間有一個回調低點 (Neckline)
        3. H1 和 H2 價格差異在容許範圍內

        【看跌訊號】
        價格跌破頸線時確認形態完成
        """
        patterns = []

        if len(highs) < 2:
            return patterns

        for i in range(len(highs) - 1):
            h1_idx, h1_price = highs[i]
            h2_idx, h2_price = highs[i + 1]

            if h2_idx - h1_idx < 10:
                continue

            price_diff = abs(h1_price - h2_price) / max(h1_price, h2_price)
            if price_diff > self.tolerance:
                continue

            middle_lows = df.iloc[h1_idx:h2_idx]['Low']
            if len(middle_lows) < 3:
                continue

            neckline_idx = middle_lows.idxmin()
            neckline_price = middle_lows.min()

            avg_high = (h1_price + h2_price) / 2
            depth = (avg_high - neckline_price) / avg_high

            if depth < 0.03:
                continue

            patterns.append({
                'type': 'M頭',
                'direction': 'bearish',
                'h1': (h1_idx, h1_price),
                'h2': (h2_idx, h2_price),
                'neckline': (neckline_idx, neckline_price),
                'depth': depth
            })

        return patterns

    def _detect_head_shoulders_bottom(self, df: pd.DataFrame, lows: List) -> List[Dict]:
        """
        識別頭肩底形態

        【形態特徵】
        1. 三個低點: 左肩 > 頭部 < 右肩
        2. 左肩和右肩價格接近
        3. 頭部是最低點

        【看漲訊號】
        突破頸線確認反轉
        """
        patterns = []

        if len(lows) < 3:
            return patterns

        for i in range(len(lows) - 2):
            ls_idx, ls_price = lows[i]      # 左肩
            head_idx, head_price = lows[i + 1]  # 頭部
            rs_idx, rs_price = lows[i + 2]  # 右肩

            # 條件: 頭部低於兩肩
            if not (head_price < ls_price and head_price < rs_price):
                continue

            # 條件: 兩肩價格接近 (5% 容許度)
            shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
            if shoulder_diff > 0.05:
                continue

            # 尋找頸線 (兩肩之間的高點連線)
            left_neckline = df.iloc[ls_idx:head_idx]['High'].max()
            right_neckline = df.iloc[head_idx:rs_idx]['High'].max()
            neckline_price = (left_neckline + right_neckline) / 2

            patterns.append({
                'type': '頭肩底',
                'direction': 'bullish',
                'left_shoulder': (ls_idx, ls_price),
                'head': (head_idx, head_price),
                'right_shoulder': (rs_idx, rs_price),
                'neckline_price': neckline_price
            })

        return patterns

    def _detect_head_shoulders_top(self, df: pd.DataFrame, highs: List) -> List[Dict]:
        """
        識別頭肩頂形態

        【形態特徵】
        1. 三個高點: 左肩 < 頭部 > 右肩
        2. 左肩和右肩價格接近
        3. 頭部是最高點

        【看跌訊號】
        跌破頸線確認反轉
        """
        patterns = []

        if len(highs) < 3:
            return patterns

        for i in range(len(highs) - 2):
            ls_idx, ls_price = highs[i]
            head_idx, head_price = highs[i + 1]
            rs_idx, rs_price = highs[i + 2]

            if not (head_price > ls_price and head_price > rs_price):
                continue

            shoulder_diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
            if shoulder_diff > 0.05:
                continue

            left_neckline = df.iloc[ls_idx:head_idx]['Low'].min()
            right_neckline = df.iloc[head_idx:rs_idx]['Low'].min()
            neckline_price = (left_neckline + right_neckline) / 2

            patterns.append({
                'type': '頭肩頂',
                'direction': 'bearish',
                'left_shoulder': (ls_idx, ls_price),
                'head': (head_idx, head_price),
                'right_shoulder': (rs_idx, rs_price),
                'neckline_price': neckline_price
            })

        return patterns

    # ========================================
    # 評分系統
    # ========================================

    def _calculate_score(
        self,
        df: pd.DataFrame,
        pattern: Dict,
        vol_ma: pd.Series,
        atr: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
        """
        計算形態綜合評分

        【評分公式】
        Total Score = Σ(分項得分 × 權重)

        分項得分 (滿分 100):
        1. 幾何標準度 (50%): 轉折點誤差越小分數越高
        2. 帶量突破 (20%): 突破時成交量相對於均量的倍數
        3. 訂單塊共振 (15%): 轉折點是否測試訂單塊
        4. 斐波納契 (15%): 轉折點是否在 Fibo 回撤位

        Returns:
            (總分, 分項得分字典)
        """
        scores = {
            'geometry': 0,
            'volume': 0,
            'order_block': 0,
            'fibonacci': 0
        }

        pattern_type = pattern['type']

        # ========================================
        # 1. 幾何標準度評分 (滿分 100)
        # ========================================
        # 根據形態類型計算轉折點的標準程度
        if pattern_type in ['W底', 'M頭']:
            # 雙底/雙頂: 評估兩個轉折點的對稱性
            if pattern_type == 'W底':
                p1, p2 = pattern['l1'][1], pattern['l2'][1]
            else:
                p1, p2 = pattern['h1'][1], pattern['h2'][1]

            # 誤差越小，分數越高
            # 公式: score = 100 - (error_pct / tolerance × 50)
            error_pct = abs(p1 - p2) / max(p1, p2)
            scores['geometry'] = max(0, 100 - (error_pct / self.tolerance * 50))

        elif pattern_type in ['頭肩底', '頭肩頂']:
            # 頭肩形態: 評估兩肩的對稱性
            ls = pattern['left_shoulder'][1]
            rs = pattern['right_shoulder'][1]
            error_pct = abs(ls - rs) / max(ls, rs)
            scores['geometry'] = max(0, 100 - (error_pct / 0.05 * 50))

        # ========================================
        # 2. 帶量突破評分 (滿分 100)
        # ========================================
        # 檢查形態完成時的成交量
        if pattern_type in ['W底', '頭肩底']:
            # 看漲形態: 檢查最近幾根 K 線的量能
            recent_vol = df['Volume'].iloc[-5:].mean()
            recent_vol_ma = vol_ma.iloc[-5:].mean()

            if recent_vol_ma > 0:
                vol_ratio = recent_vol / recent_vol_ma
                # 量能倍數 >= 1.5 得滿分，1.0 得 50 分
                scores['volume'] = min(100, max(0, (vol_ratio - 1.0) * 100))

        elif pattern_type in ['M頭', '頭肩頂']:
            recent_vol = df['Volume'].iloc[-5:].mean()
            recent_vol_ma = vol_ma.iloc[-5:].mean()

            if recent_vol_ma > 0:
                vol_ratio = recent_vol / recent_vol_ma
                scores['volume'] = min(100, max(0, (vol_ratio - 1.0) * 100))

        # ========================================
        # 3. 訂單塊共振評分 (滿分 100)
        # ========================================
        # 檢查轉折點是否精準測試訂單塊區域
        if pattern_type == 'W底':
            pivot_idx = pattern['l2'][0] if isinstance(pattern['l2'][0], int) else df.index.get_loc(pattern['l2'][0])
            pivot_price = pattern['l2'][1]
        elif pattern_type == 'M頭':
            pivot_idx = pattern['h2'][0] if isinstance(pattern['h2'][0], int) else df.index.get_loc(pattern['h2'][0])
            pivot_price = pattern['h2'][1]
        elif pattern_type == '頭肩底':
            pivot_idx = pattern['head'][0] if isinstance(pattern['head'][0], int) else df.index.get_loc(pattern['head'][0])
            pivot_price = pattern['head'][1]
        elif pattern_type == '頭肩頂':
            pivot_idx = pattern['head'][0] if isinstance(pattern['head'][0], int) else df.index.get_loc(pattern['head'][0])
            pivot_price = pattern['head'][1]
        else:
            pivot_idx = 0
            pivot_price = 0

        order_block = self._find_order_blocks(df, pivot_idx)
        if order_block:
            ob_high, ob_low = order_block['high'], order_block['low']
            # 檢查價格是否在訂單塊範圍內
            if ob_low <= pivot_price <= ob_high:
                scores['order_block'] = 100
            elif self._is_near_level(pivot_price, ob_high, 0.02) or \
                 self._is_near_level(pivot_price, ob_low, 0.02):
                scores['order_block'] = 70

        # ========================================
        # 4. 斐波納契共振評分 (滿分 100)
        # ========================================
        # 計算近期波段的 Fibo 回撤位
        recent_high = df['High'].iloc[-50:].max()
        recent_low = df['Low'].iloc[-50:].min()
        fibo_levels = self._calculate_fibonacci(recent_high, recent_low)

        # 檢查轉折點是否在關鍵 Fibo 位置 (0.5 或 0.618)
        key_fibs = [fibo_levels['0.500'], fibo_levels['0.618']]
        for fib_level in key_fibs:
            if self._is_near_level(pivot_price, fib_level, 0.02):
                scores['fibonacci'] = 100
                break

        # 次要 Fibo 位置給予較低分數
        if scores['fibonacci'] == 0:
            for key, fib_level in fibo_levels.items():
                if self._is_near_level(pivot_price, fib_level, 0.02):
                    scores['fibonacci'] = 60
                    break

        # ========================================
        # 計算加權總分
        # ========================================
        # 公式: Total = Σ(Score_i × Weight_i)
        total_score = (
            scores['geometry'] * SCORING_WEIGHTS['geometry'] +
            scores['volume'] * SCORING_WEIGHTS['volume'] +
            scores['order_block'] * SCORING_WEIGHTS['order_block'] +
            scores['fibonacci'] * SCORING_WEIGHTS['fibonacci']
        )

        return total_score, scores

    def detect_and_score(self, df: pd.DataFrame, ticker: str = "", market: str = "") -> List[PatternResult]:
        """
        主要方法：偵測形態並評分

        【執行流程】
        1. 計算技術指標 (ATR, Volume MA)
        2. 尋找轉折點 (Pivot High/Low)
        3. 識別各類形態
        4. 計算評分
        5. 篩選高品質形態 (Score >= 75, 最近 50 根 K 線內)

        Args:
            df: 標準化的 OHLCV DataFrame
            ticker: 股票代號 (用於結果標記)
            market: 市場 (用於結果標記)

        Returns:
            PatternResult 列表 (只包含高分形態)
        """
        if df is None or len(df) < 50:
            return []

        results = []

        try:
            # 計算技術指標
            atr = self._calculate_atr(df)
            vol_ma = self._calculate_volume_ma(df)

            # 尋找轉折點
            pivot_highs, pivot_lows = self._find_pivots(df)

            # 識別所有形態
            all_patterns = []
            all_patterns.extend(self._detect_double_bottom(df, pivot_lows))
            all_patterns.extend(self._detect_double_top(df, pivot_highs))
            all_patterns.extend(self._detect_head_shoulders_bottom(df, pivot_lows))
            all_patterns.extend(self._detect_head_shoulders_top(df, pivot_highs))

            # 評分並篩選
            for pattern in all_patterns:
                score, breakdown = self._calculate_score(df, pattern, vol_ma, atr)

                # 篩選條件1: 總分 >= 75
                if score < SCORE_THRESHOLD:
                    continue

                # 取得形態結束索引
                if pattern['type'] in ['W底', 'M頭']:
                    end_idx = pattern.get('l2', pattern.get('h2'))[0]
                else:
                    end_idx = pattern.get('right_shoulder', pattern.get('head'))[0]

                # 轉換為整數索引
                if not isinstance(end_idx, int):
                    try:
                        end_idx = df.index.get_loc(end_idx)
                    except:
                        end_idx = len(df) - 1

                # 篩選條件2: 訊號在最近 50 根 K 線內
                if len(df) - end_idx > 50:
                    continue

                # 建立結果物件
                signal_date = df.index[end_idx] if end_idx < len(df) else df.index[-1]

                result = PatternResult(
                    ticker=ticker,
                    market=market,
                    pattern_name=pattern['type'],
                    direction=pattern['direction'],
                    score=round(score, 1),
                    current_price=round(df['Close'].iloc[-1], 2),
                    signal_date=signal_date,
                    key_levels={
                        'neckline': pattern.get('neckline', (0, 0))[1] if isinstance(pattern.get('neckline'), tuple) else pattern.get('neckline_price', 0)
                    },
                    score_breakdown=breakdown
                )
                results.append(result)

        except Exception as e:
            print(f"  [引擎錯誤] {ticker}: {e}")

        return results


# ============================================================
# [Task 3] 掃描控制器 (MarketScanner Class)
# ============================================================

class MarketScanner:
    """
    市場掃描控制器

    功能：
    1. 批次掃描多檔股票
    2. 錯誤處理 (單一股票錯誤不中斷掃描)
    3. 進度顯示 (tqdm)
    """

    def __init__(self, data_manager: DataManager, pattern_engine: PatternEngine):
        """
        初始化掃描器

        Args:
            data_manager: 數據管理器實例
            pattern_engine: 形態引擎實例
        """
        self.data_manager = data_manager
        self.pattern_engine = pattern_engine

    def scan_market(self, market: str = 'TW', tickers: List[str] = None) -> List[PatternResult]:
        """
        掃描指定市場

        Args:
            market: 市場代碼 ('TW' 或 'US')
            tickers: 股票清單 (若為 None 則使用預設清單)

        Returns:
            所有高分形態結果列表
        """
        if tickers is None:
            tickers = WATCHLIST.get(market.upper(), [])

        if not tickers:
            print(f"[警告] {market} 市場沒有股票清單")
            return []

        results = []

        print(f"\n{'='*50}")
        print(f"開始掃描 {market} 市場 ({len(tickers)} 檔股票)")
        print(f"{'='*50}")

        # 使用 tqdm 顯示進度
        for ticker in tqdm(tickers, desc=f"掃描 {market}"):
            try:
                # 取得數據
                df = self.data_manager.get_stock_data(ticker, market)

                if df is None or len(df) < 50:
                    continue

                # 偵測形態
                patterns = self.pattern_engine.detect_and_score(df, ticker, market)
                results.extend(patterns)

            except Exception as e:
                # 錯誤處理: 記錄但不中斷
                print(f"  [跳過] {ticker}: {e}")
                continue

        print(f"\n掃描完成！找到 {len(results)} 個高品質形態")

        return results

    def scan_all_markets(self) -> List[PatternResult]:
        """掃描所有市場"""
        all_results = []

        for market in ['TW', 'US']:
            results = self.scan_market(market)
            all_results.extend(results)

        return all_results

    def results_to_dataframe(self, results: List[PatternResult]) -> pd.DataFrame:
        """將結果轉換為 DataFrame"""
        if not results:
            return pd.DataFrame(columns=['代號', '市場', '形態', '方向', '分數', '現價', '訊號日期'])

        data = []
        for r in results:
            direction_str = '🟢 看漲' if r.direction == 'bullish' else '🔴 看跌'
            data.append({
                '代號': r.ticker,
                '市場': r.market,
                '形態': r.pattern_name,
                '方向': direction_str,
                '分數': r.score,
                '現價': r.current_price,
                '訊號日期': r.signal_date.strftime('%Y-%m-%d') if hasattr(r.signal_date, 'strftime') else str(r.signal_date)
            })

        df = pd.DataFrame(data)
        df = df.sort_values('分數', ascending=False).reset_index(drop=True)

        return df


# ============================================================
# [Task 4] 視覺化模組 (Visualization)
# ============================================================

class PatternVisualizer:
    """形態視覺化器"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def plot_pattern(
        self,
        ticker: str,
        market: str,
        pattern_result: PatternResult = None
    ) -> go.Figure:
        """
        繪製股票 K 線圖與形態標註

        Args:
            ticker: 股票代號
            market: 市場
            pattern_result: 形態結果 (可選)

        Returns:
            Plotly Figure 物件
        """
        # 取得數據
        df = self.data_manager.get_stock_data(ticker, market)

        if df is None:
            print(f"無法取得 {ticker} 數據")
            return None

        # 建立子圖 (K線 + 成交量)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=[f'{ticker} ({market})', '成交量']
        )

        # K 線圖
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='K線',
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )

        # 成交量柱狀圖
        colors = ['red' if c >= o else 'green'
                  for c, o in zip(df['Close'], df['Open'])]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name='成交量',
                opacity=0.7
            ),
            row=2, col=1
        )

        # 如果有形態結果，標註形態區域
        if pattern_result:
            neckline = pattern_result.key_levels.get('neckline', 0)

            if neckline > 0:
                # 繪製頸線
                fig.add_hline(
                    y=neckline,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"頸線 {neckline:.2f}",
                    row=1, col=1
                )

            # 添加形態標註文字
            direction_color = 'green' if pattern_result.direction == 'bullish' else 'red'
            fig.add_annotation(
                x=df.index[-1],
                y=df['High'].max(),
                text=f"{pattern_result.pattern_name}<br>分數: {pattern_result.score}",
                showarrow=True,
                arrowhead=1,
                font=dict(color=direction_color, size=14),
                bgcolor='white',
                bordercolor=direction_color
            )

        # 圖表設定
        fig.update_layout(
            title=f'{ticker} 技術形態分析',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="價格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)

        return fig


# ============================================================
# [Task 4] 互動式儀表板 (Dashboard UI)
# ============================================================

class Dashboard:
    """
    互動式儀表板

    使用 ipywidgets 建立 Colab 互動介面
    """

    def __init__(self):
        self.data_manager = None
        self.pattern_engine = PatternEngine()
        self.scanner = None
        self.visualizer = None
        self.results = []
        self.results_df = None

        # UI 元件
        self.finmind_input = None
        self.tiingo_input = None
        self.market_dropdown = None
        self.scan_button = None
        self.output_area = None
        self.result_dropdown = None
        self.chart_output = None

    def _create_widgets(self):
        """建立 UI 元件"""
        if not HAS_WIDGETS:
            print("[錯誤] ipywidgets 未安裝，無法使用互動介面")
            return False

        # API Key 輸入
        self.finmind_input = widgets.Text(
            value='',
            placeholder='輸入 FinMind Token (台股用)',
            description='FinMind:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )

        self.tiingo_input = widgets.Text(
            value='',
            placeholder='輸入 Tiingo API Key (美股用)',
            description='Tiingo:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )

        # 市場選擇
        self.market_dropdown = widgets.Dropdown(
            options=[('台股 (TW)', 'TW'), ('美股 (US)', 'US'), ('全部市場', 'ALL')],
            value='TW',
            description='市場:',
            style={'description_width': '80px'}
        )

        # 掃描按鈕
        self.scan_button = widgets.Button(
            description='🚀 開始掃描',
            button_style='success',
            layout=widgets.Layout(width='150px', height='40px')
        )
        self.scan_button.on_click(self._on_scan_click)

        # 輸出區域
        self.output_area = widgets.Output(
            layout=widgets.Layout(border='1px solid #ccc', min_height='200px')
        )

        # 結果選擇下拉選單
        self.result_dropdown = widgets.Dropdown(
            options=[],
            description='查看:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='300px')
        )
        self.result_dropdown.observe(self._on_result_select, names='value')

        # 圖表輸出
        self.chart_output = widgets.Output()

        return True

    def _on_scan_click(self, button):
        """掃描按鈕點擊事件"""
        with self.output_area:
            clear_output()

            # 初始化數據管理器
            self.data_manager = DataManager(
                finmind_token=self.finmind_input.value,
                tiingo_key=self.tiingo_input.value
            )

            self.scanner = MarketScanner(self.data_manager, self.pattern_engine)
            self.visualizer = PatternVisualizer(self.data_manager)

            # 執行掃描
            market = self.market_dropdown.value

            if market == 'ALL':
                self.results = self.scanner.scan_all_markets()
            else:
                self.results = self.scanner.scan_market(market)

            # 顯示結果
            if self.results:
                self.results_df = self.scanner.results_to_dataframe(self.results)

                print("\n" + "="*60)
                print("📊 掃描結果 (分數 >= 75)")
                print("="*60)
                display(self.results_df)

                # 更新結果下拉選單
                options = [(f"{r.ticker} - {r.pattern_name} ({r.score}分)", i)
                           for i, r in enumerate(self.results)]
                self.result_dropdown.options = options

                if options:
                    self.result_dropdown.value = 0
            else:
                print("\n⚠️ 未找到符合條件的形態 (分數 >= 75)")

    def _on_result_select(self, change):
        """結果選擇事件"""
        if change['new'] is None or not self.results:
            return

        idx = change['new']
        result = self.results[idx]

        with self.chart_output:
            clear_output()
            fig = self.visualizer.plot_pattern(result.ticker, result.market, result)
            if fig:
                fig.show()

    def display(self):
        """顯示儀表板"""
        if not self._create_widgets():
            return

        # 標題
        title = widgets.HTML(
            value="""
            <h2 style='color: #2E86AB; margin-bottom: 20px;'>
                🔍 Omni-Scanner Pro - 市場形態掃描系統
            </h2>
            <p style='color: #666;'>
                支援台股 (FinMind) 與美股 (Tiingo) | 12小時智慧快取 |
                評分門檻: 75分 | 權重: 幾何50% + 量能20% + OB共振15% + Fibo15%
            </p>
            """
        )

        # 設定區塊
        settings_box = widgets.VBox([
            widgets.HTML("<h4>📝 API 設定</h4>"),
            self.finmind_input,
            self.tiingo_input,
            widgets.HBox([self.market_dropdown, self.scan_button])
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0'))

        # 結果區塊
        result_box = widgets.VBox([
            widgets.HTML("<h4>📈 掃描結果</h4>"),
            self.output_area,
            widgets.HBox([
                widgets.HTML("<b>詳細圖表:</b>"),
                self.result_dropdown
            ]),
            self.chart_output
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='10px 0'))

        # 組合顯示
        dashboard = widgets.VBox([title, settings_box, result_box])
        display(dashboard)


# ============================================================
# [Task 5] 命令列模式 (CLI Mode)
# ============================================================

def run_cli_mode():
    """
    命令列模式 (當 ipywidgets 不可用時)
    """
    print("\n" + "="*60)
    print("🔍 Omni-Scanner Pro - 命令列模式")
    print("="*60)

    # 輸入 API Key
    print("\n請輸入 API Keys (可留空跳過):")
    finmind_token = input("FinMind Token (台股): ").strip()
    tiingo_key = input("Tiingo API Key (美股): ").strip()

    # 選擇市場
    print("\n選擇市場:")
    print("1. 台股 (TW)")
    print("2. 美股 (US)")
    print("3. 全部")

    choice = input("請選擇 (1/2/3): ").strip()

    market_map = {'1': 'TW', '2': 'US', '3': 'ALL'}
    market = market_map.get(choice, 'TW')

    # 初始化元件
    data_manager = DataManager(finmind_token, tiingo_key)
    pattern_engine = PatternEngine()
    scanner = MarketScanner(data_manager, pattern_engine)

    # 執行掃描
    if market == 'ALL':
        results = scanner.scan_all_markets()
    else:
        results = scanner.scan_market(market)

    # 顯示結果
    if results:
        df = scanner.results_to_dataframe(results)
        print("\n" + "="*60)
        print("📊 掃描結果")
        print("="*60)
        print(df.to_string())
    else:
        print("\n⚠️ 未找到符合條件的形態")


# ============================================================
# [Task 5] 主程式入口 (Main Entry)
# ============================================================

def main():
    """
    主程式入口

    執行流程:
    1. 環境檢查
    2. 根據環境選擇模式 (Colab Dashboard / CLI)
    """
    print("="*60)
    print("🚀 Omni-Scanner Pro v1.0")
    print("="*60)

    # 環境檢查
    print("\n📋 環境檢查:")
    print(f"  - FinMind (台股): {'✅' if HAS_FINMIND else '❌'}")
    print(f"  - yfinance (美股): {'✅' if HAS_YFINANCE else '❌'}")
    print(f"  - pandas_ta: {'✅' if HAS_PANDAS_TA else '⚠️ (使用內建)'}")
    print(f"  - ipywidgets: {'✅' if HAS_WIDGETS else '❌ (CLI 模式)'}")

    # 根據環境選擇模式
    if HAS_WIDGETS:
        try:
            # 嘗試啟動 Dashboard
            dashboard = Dashboard()
            dashboard.display()
        except Exception as e:
            print(f"\n[警告] Dashboard 啟動失敗: {e}")
            print("切換至命令列模式...\n")
            run_cli_mode()
    else:
        run_cli_mode()


# ============================================================
# 程式進入點
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# 快速測試函數 (供開發使用)
# ============================================================

def quick_test(ticker: str = "2330", market: str = "TW"):
    """
    快速測試單一股票

    Usage:
        quick_test("2330", "TW")
        quick_test("AAPL", "US")
    """
    print(f"\n🧪 快速測試: {ticker} ({market})")

    # 使用空 Token (依賴快取或公開 API)
    dm = DataManager()
    engine = PatternEngine()
    viz = PatternVisualizer(dm)

    # 取得數據
    df = dm.get_stock_data(ticker, market)

    if df is not None:
        print(f"✅ 數據取得成功: {len(df)} 筆")

        # 偵測形態
        results = engine.detect_and_score(df, ticker, market)

        if results:
            print(f"✅ 找到 {len(results)} 個高分形態:")
            for r in results:
                print(f"   - {r.pattern_name}: {r.score} 分")
        else:
            print("⚠️ 未找到高分形態 (分數 < 75)")

        # 繪圖
        fig = viz.plot_pattern(ticker, market, results[0] if results else None)
        if fig:
            fig.show()
    else:
        print("❌ 數據取得失敗")
