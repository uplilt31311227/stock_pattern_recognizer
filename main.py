# -*- coding: utf-8 -*-
"""
================================================================================
智慧型 K 線形態辨識系統 v2.0 - 本地部署版
================================================================================
預判型系統 - 在形態完成前提前識別潛在機會

使用方式:
    uv run python main.py                    # 互動模式
    uv run python main.py --market US        # 掃描美股
    uv run python main.py --market TW        # 掃描台股
    uv run python main.py --mode full        # 全市場掃描
    uv run python main.py --list             # 查看歷史結果

================================================================================
"""

import os
import sys
import time
import json
import math
import csv
import io
import zipfile
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

# yfinance
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

warnings.filterwarnings('ignore')

# ============================================================
# Rich Console
# ============================================================
console = Console()

# ============================================================
# 全域設定
# ============================================================
CACHE_DIR = Path('./cache')
RESULTS_DIR = Path('./results')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_EXPIRY_HOURS = 12

# ZigZag 參數
ZIGZAG_THRESHOLD = 0.03
ZIGZAG_MIN_BARS = 3

# 形態參數
TOLERANCE = 0.03
NECKLINE_MIN_HEIGHT = 0.08
MIN_PATTERN_BARS = 15
HS_SYMMETRY_TOL = 0.10
NECKLINE_MAX_SLOPE = 30

# 預判觸發參數
TRIGGER_THRESHOLD = 0.02

# 評分參數
QUALITY_THRESHOLD = 65
WEIGHT_VOLUME = 0.40
WEIGHT_TREND = 0.30
WEIGHT_MORPHOLOGY = 0.30

# 時框設定
TIMEFRAMES = ['1M', '1W', '1D']
TF_LABELS = {'1D': '日線', '1W': '週線', '1M': '月線'}

# API 速率控制
TIINGO_RATE_DELAY = 1.5
FINMIND_RATE_DELAY = 0.35

# pandas 版本
_PD_VER = tuple(int(x) for x in pd.__version__.split('.')[:2])
_MONTH_RULE = 'ME' if _PD_VER >= (2, 1) else 'M'

# 精選清單
QUICK_LIST = {
    'TW': [
        '2330','2454','2317','2382','2303','3711','2379','3034',
        '6770','2408','3231','2354','2395','3008','2474','6505',
        '2881','2882','2884','2886','2891','2892','5880','2885',
        '1301','1303','1326','2603','2609','2615','2002','1402',
        '6446','4904','3045','2912','9910','2207','1216','2105',
    ],
    'US': [
        'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX',
        'AMD','INTC','AVGO','QCOM','MU','MRVL','AMAT','LRCX',
        'KLAC','ON','TXN','ADI','CRM','ORCL','ADBE','NOW',
        'SNOW','PLTR','PANW','CRWD','DDOG','NET','ZS','MNDY',
        'SHOP','PINS','SNAP','ROKU','PYPL','COIN','UBER',
        'LLY','UNH','JNJ','ABBV','MRK','PFE','TMO','ISRG',
        'JPM','BAC','GS','MS','V','MA','AXP','BLK',
        'XOM','CVX','COP','SLB','FCX','NEM',
        'WMT','COST','HD','NKE','SBUX','MCD','DIS','ABNB',
        'SPY','QQQ','IWM','XLF','XLE','XLK','SOXX','ARKK',
    ],
}


# ============================================================
# 數據結構
# ============================================================
@dataclass
class Pivot:
    idx: int
    price: float
    type: str
    date: str


@dataclass
class Pattern:
    type: str
    direction: str
    pivots: List[Pivot]
    neckline: float
    end_idx: int


@dataclass
class PatternSignal:
    symbol: str
    pattern_type: str
    signal_type: str
    timestamp: str
    timeframe: str
    quality_score: int
    entry_zone: float
    stop_loss: float
    target_neckline: float
    measured_move_target: float
    risk_reward_ratio: float
    pattern_details: Dict = field(default_factory=dict)
    score_breakdown: Dict = field(default_factory=dict)


# ============================================================
# 設定管理
# ============================================================
class Config:
    """設定管理器"""
    _config_file = Path('./config.json')

    @staticmethod
    def load() -> Dict:
        try:
            return json.loads(Config._config_file.read_text())
        except Exception:
            return {}

    @staticmethod
    def save(cfg: Dict):
        Config._config_file.write_text(json.dumps(cfg, indent=2))

    @staticmethod
    def get_api_keys() -> Tuple[str, str]:
        cfg = Config.load()
        return cfg.get('finmind_token', ''), cfg.get('tiingo_key', '')

    @staticmethod
    def set_api_keys(finmind_token: str = '', tiingo_key: str = ''):
        cfg = Config.load()
        if finmind_token:
            cfg['finmind_token'] = finmind_token
        if tiingo_key:
            cfg['tiingo_key'] = tiingo_key
        Config.save(cfg)


# ============================================================
# Ticker Registry
# ============================================================
class TickerRegistry:
    _cache_file = CACHE_DIR / '_ticker_registry.json'

    @staticmethod
    def _load():
        try:
            return json.loads(TickerRegistry._cache_file.read_text())
        except Exception:
            return {}

    @staticmethod
    def _save(d):
        TickerRegistry._cache_file.write_text(json.dumps(d, ensure_ascii=False))

    @staticmethod
    def fetch_us(tiingo_key: str = '') -> List[str]:
        cache = TickerRegistry._load()
        if cache.get('US') and cache.get('US_ts'):
            try:
                age = (datetime.now() - datetime.fromisoformat(cache['US_ts'])).total_seconds()
                if age < 86400:
                    return cache['US']
            except Exception:
                pass

        console.print('[yellow]下載美股清單中...[/yellow]')
        url = 'https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip'
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f'Tiingo 清單下載失敗: HTTP {r.status_code}')

        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f)))

        cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        tickers = sorted(set(
            row['ticker'] for row in rows
            if row.get('exchange') in ('NYSE', 'NASDAQ')
            and row.get('assetType') == 'Stock'
            and row.get('endDate', '') >= cutoff
        ))

        cache['US'] = tickers
        cache['US_ts'] = datetime.now().isoformat()
        TickerRegistry._save(cache)
        return tickers

    @staticmethod
    def fetch_tw(finmind_token: str = '') -> List[str]:
        cache = TickerRegistry._load()
        if cache.get('TW') and cache.get('TW_ts'):
            try:
                age = (datetime.now() - datetime.fromisoformat(cache['TW_ts'])).total_seconds()
                if age < 86400:
                    return cache['TW']
            except Exception:
                pass

        console.print('[yellow]下載台股清單中...[/yellow]')
        params = {'dataset': 'TaiwanStockInfo'}
        if finmind_token:
            params['token'] = finmind_token
        r = requests.get('https://api.finmindtrade.com/api/v4/data', params=params, timeout=30)
        data = r.json()
        if data.get('status') != 200:
            raise RuntimeError(f'FinMind 錯誤: {data.get("msg")}')

        tickers = sorted(set(
            s['stock_id'] for s in data['data']
            if s.get('type') in ('twse', 'tpex')
        ))

        cache['TW'] = tickers
        cache['TW_ts'] = datetime.now().isoformat()
        TickerRegistry._save(cache)
        return tickers


# ============================================================
# DataManager
# ============================================================
class DataManager:
    def __init__(self, finmind_token='', tiingo_key=''):
        self.fm_token = finmind_token
        self.tg_key = tiingo_key
        self.session = requests.Session()
        self._last_us = 0
        self._last_tw = 0
        self._backoff = 0
        self.stats = {'api_us': 0, 'api_tw': 0, 'yf_fallback': 0, 'cache': 0, 'err': 0}

    def _cache_path(self, ticker, market):
        return CACHE_DIR / f"{market}_{ticker.replace('.','_')}.csv"

    def _cache_ok(self, p):
        if not p.exists():
            return False
        age = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds()
        return age < CACHE_EXPIRY_HOURS * 3600

    def _wait_us(self):
        delay = TIINGO_RATE_DELAY + self._backoff
        gap = time.time() - self._last_us
        if gap < delay:
            time.sleep(delay - gap)
        self._last_us = time.time()

    def _wait_tw(self):
        gap = time.time() - self._last_tw
        if gap < FINMIND_RATE_DELAY:
            time.sleep(FINMIND_RATE_DELAY - gap)
        self._last_tw = time.time()

    def _fetch_us(self, ticker):
        self._wait_us()
        end = datetime.now()
        start = end - timedelta(days=2000)
        url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
        params = {
            'startDate': start.strftime('%Y-%m-%d'),
            'endDate': end.strftime('%Y-%m-%d'),
            'token': self.tg_key
        }
        r = self.session.get(url, params=params, timeout=15)
        if r.status_code == 429:
            self._backoff = min(30, self._backoff + 3)
            time.sleep(self._backoff)
            self._last_us = time.time()
            r = self.session.get(url, params=params, timeout=15)
        if r.status_code != 200:
            raise ValueError(f'Tiingo HTTP {r.status_code}')
        data = r.json()
        if not data:
            raise ValueError('Tiingo 無數據')
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.set_index('date')
        adj_cols = ['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']
        df = df[adj_cols].copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.stats['api_us'] += 1
        self._backoff = max(0, self._backoff - 0.5)
        return df

    def _fetch_us_yf(self, ticker):
        if not HAS_YF:
            raise ImportError('yfinance 未安裝')
        end = datetime.now()
        start = end - timedelta(days=2000)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError('yfinance 無數據')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self.stats['yf_fallback'] += 1
        return df

    def _fetch_tw(self, ticker):
        self._wait_tw()
        end = datetime.now()
        start = end - timedelta(days=2000)
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': ticker,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
        }
        if self.fm_token:
            params['token'] = self.fm_token
        r = self.session.get('https://api.finmindtrade.com/api/v4/data', params=params, timeout=15)
        data = r.json()
        if data.get('status') != 200:
            raise ValueError(f'FinMind: {data.get("msg","")}')
        rows = data['data']
        if not rows:
            raise ValueError('FinMind 無數據')
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        self.stats['api_tw'] += 1
        return df

    def _standardize(self, df):
        col_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
            'max': 'High', 'min': 'Low', 'Trading_Volume': 'Volume',
        }
        df = df.rename(columns=col_map)
        for target in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if target not in df.columns:
                for c in df.columns:
                    if c.lower() == target.lower():
                        df = df.rename(columns={c: target})
                        break
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep].copy()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.sort_index().dropna()

    @staticmethod
    def resample(df, timeframe='1D'):
        if timeframe == '1D' or df is None or df.empty:
            return df
        rule = {'1W': 'W', '1M': _MONTH_RULE}[timeframe]
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        return df.resample(rule).agg(agg).dropna()

    def get(self, ticker, market='TW'):
        cp = self._cache_path(ticker, market)
        try:
            if self._cache_ok(cp):
                df = pd.read_csv(cp, index_col=0, parse_dates=True)
                self.stats['cache'] += 1
                return self._standardize(df)
            if market.upper() == 'TW':
                df = self._fetch_tw(ticker)
            else:
                try:
                    df = self._fetch_us(ticker)
                except Exception:
                    df = self._fetch_us_yf(ticker)
            df = self._standardize(df)
            df.to_csv(cp)
            return df
        except Exception:
            self.stats['err'] += 1
            return None


# ============================================================
# 數據前處理
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df[df['Volume'] > 0].copy()
    df = df[df['High'] != df['Low']].copy()
    return df


# ============================================================
# ZigZag 演算法
# ============================================================
def zigzag(df: pd.DataFrame, threshold: float = ZIGZAG_THRESHOLD,
           min_bars: int = ZIGZAG_MIN_BARS) -> List[Pivot]:
    if df is None or len(df) < min_bars * 2:
        return []

    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    n = len(df)

    pivots = []

    if highs[0] >= lows[0]:
        last_type = 'PEAK'
    else:
        last_type = 'VALLEY'

    potential_high_idx = 0
    potential_high = highs[0]
    potential_low_idx = 0
    potential_low = lows[0]

    for i in range(1, n):
        if highs[i] > potential_high:
            potential_high = highs[i]
            potential_high_idx = i

        if lows[i] < potential_low:
            potential_low = lows[i]
            potential_low_idx = i

        if last_type == 'PEAK':
            if potential_high > 0 and (highs[i] - potential_low) / potential_low >= threshold:
                if i - potential_low_idx >= min_bars:
                    date_str = str(dates[potential_low_idx].date()) if hasattr(dates[potential_low_idx], 'date') else str(dates[potential_low_idx])
                    pivots.append(Pivot(potential_low_idx, potential_low, 'VALLEY', date_str))
                    last_type = 'VALLEY'
                    potential_high = highs[i]
                    potential_high_idx = i
        else:
            if potential_low > 0 and (potential_high - lows[i]) / potential_high >= threshold:
                if i - potential_high_idx >= min_bars:
                    date_str = str(dates[potential_high_idx].date()) if hasattr(dates[potential_high_idx], 'date') else str(dates[potential_high_idx])
                    pivots.append(Pivot(potential_high_idx, potential_high, 'PEAK', date_str))
                    last_type = 'PEAK'
                    potential_low = lows[i]
                    potential_low_idx = i

    return list(reversed(pivots))


# ============================================================
# 形態識別
# ============================================================
class PatternDetector:

    @staticmethod
    def detect_double_bottom(pivots: List[Pivot], df: pd.DataFrame) -> Optional[Pattern]:
        if len(pivots) < 4:
            return None

        p0, p1, p2, p3 = pivots[0], pivots[1], pivots[2], pivots[3]

        if p3.type != 'VALLEY' or p1.type != 'VALLEY':
            return None
        if p2.type != 'PEAK':
            return None

        left_bottom = p3.price
        right_bottom = p1.price
        neckline = p2.price

        if abs(left_bottom - right_bottom) / left_bottom > TOLERANCE:
            return None

        avg_bottom = (left_bottom + right_bottom) / 2
        if (neckline - avg_bottom) / avg_bottom < NECKLINE_MIN_HEIGHT:
            return None

        if p1.idx - p3.idx < MIN_PATTERN_BARS:
            return None

        return Pattern(
            type='DOUBLE_BOTTOM',
            direction='bullish',
            pivots=[p3, p2, p1, p0] if p0 else [p3, p2, p1],
            neckline=neckline,
            end_idx=p1.idx
        )

    @staticmethod
    def detect_double_top(pivots: List[Pivot], df: pd.DataFrame) -> Optional[Pattern]:
        if len(pivots) < 4:
            return None

        p0, p1, p2, p3 = pivots[0], pivots[1], pivots[2], pivots[3]

        if p3.type != 'PEAK' or p1.type != 'PEAK':
            return None
        if p2.type != 'VALLEY':
            return None

        left_top = p3.price
        right_top = p1.price
        neckline = p2.price

        if abs(left_top - right_top) / left_top > TOLERANCE:
            return None

        avg_top = (left_top + right_top) / 2
        if (avg_top - neckline) / avg_top < NECKLINE_MIN_HEIGHT:
            return None

        if p1.idx - p3.idx < MIN_PATTERN_BARS:
            return None

        return Pattern(
            type='DOUBLE_TOP',
            direction='bearish',
            pivots=[p3, p2, p1, p0] if p0 else [p3, p2, p1],
            neckline=neckline,
            end_idx=p1.idx
        )

    @staticmethod
    def detect_hs_bottom(pivots: List[Pivot], df: pd.DataFrame) -> Optional[Pattern]:
        if len(pivots) < 5:
            return None

        p0, p1, p2, p3, p4 = pivots[0], pivots[1], pivots[2], pivots[3], pivots[4]

        if p4.type != 'VALLEY' or p2.type != 'VALLEY' or p0.type != 'VALLEY':
            return None
        if p3.type != 'PEAK' or p1.type != 'PEAK':
            return None

        left_shoulder = p4.price
        head = p2.price
        right_shoulder = p0.price

        if not (head < left_shoulder and head < right_shoulder):
            return None

        if abs(left_shoulder - right_shoulder) / head > HS_SYMMETRY_TOL:
            return None

        left_time = p2.idx - p4.idx
        right_time = p0.idx - p2.idx
        if right_time < left_time * 0.3:
            return None

        neckline_left = p3.price
        neckline_right = p1.price
        time_diff = p1.idx - p3.idx
        if time_diff > 0:
            price_diff = abs(neckline_right - neckline_left)
            avg_price = (neckline_left + neckline_right) / 2
            slope_ratio = price_diff / avg_price / time_diff * 100
            slope_deg = math.degrees(math.atan(slope_ratio))
            if abs(slope_deg) > NECKLINE_MAX_SLOPE:
                return None

        neckline = (neckline_left + neckline_right) / 2

        return Pattern(
            type='HS_BOTTOM',
            direction='bullish',
            pivots=[p4, p3, p2, p1, p0],
            neckline=neckline,
            end_idx=p0.idx
        )

    @staticmethod
    def detect_hs_top(pivots: List[Pivot], df: pd.DataFrame) -> Optional[Pattern]:
        if len(pivots) < 5:
            return None

        p0, p1, p2, p3, p4 = pivots[0], pivots[1], pivots[2], pivots[3], pivots[4]

        if p4.type != 'PEAK' or p2.type != 'PEAK' or p0.type != 'PEAK':
            return None
        if p3.type != 'VALLEY' or p1.type != 'VALLEY':
            return None

        left_shoulder = p4.price
        head = p2.price
        right_shoulder = p0.price

        if not (head > left_shoulder and head > right_shoulder):
            return None

        if abs(left_shoulder - right_shoulder) / head > HS_SYMMETRY_TOL:
            return None

        left_time = p2.idx - p4.idx
        right_time = p0.idx - p2.idx
        if right_time < left_time * 0.3:
            return None

        neckline_left = p3.price
        neckline_right = p1.price
        time_diff = p1.idx - p3.idx
        if time_diff > 0:
            price_diff = abs(neckline_right - neckline_left)
            avg_price = (neckline_left + neckline_right) / 2
            slope_ratio = price_diff / avg_price / time_diff * 100
            slope_deg = math.degrees(math.atan(slope_ratio))
            if abs(slope_deg) > NECKLINE_MAX_SLOPE:
                return None

        neckline = (neckline_left + neckline_right) / 2

        return Pattern(
            type='HS_TOP',
            direction='bearish',
            pivots=[p4, p3, p2, p1, p0],
            neckline=neckline,
            end_idx=p0.idx
        )

    @staticmethod
    def detect_cup_handle(df: pd.DataFrame, pivots: List[Pivot]) -> Optional[Pattern]:
        if df is None or len(df) < 60:
            return None

        lookback = min(150, len(df))
        recent = df.iloc[-lookback:]

        high_left_idx = recent['High'].iloc[:lookback//3].idxmax()
        high_right_idx = recent['High'].iloc[-lookback//3:].idxmax()

        left_idx = recent.index.get_loc(high_left_idx)
        right_idx = recent.index.get_loc(high_right_idx)

        if right_idx - left_idx < 30:
            return None

        cup_section = recent.iloc[left_idx:right_idx+1]
        if len(cup_section) < 30:
            return None

        low_idx = cup_section['Low'].idxmin()
        low_pos = cup_section.index.get_loc(low_idx)

        section_len = len(cup_section)
        if not (section_len * 0.25 < low_pos < section_len * 0.75):
            return None

        high_left = recent['High'].loc[high_left_idx]
        high_right = recent['High'].loc[high_right_idx]
        low_min = cup_section['Low'].loc[low_idx]

        cup_depth = ((high_left + high_right) / 2) - low_min

        if right_idx + 5 >= len(recent):
            return None

        handle_section = recent.iloc[right_idx:]
        if len(handle_section) < 3:
            return None

        handle_low = handle_section['Low'].min()
        handle_pullback = high_right - handle_low

        if handle_pullback > cup_depth * 0.382:
            return None

        neckline = (high_left + high_right) / 2

        date_low = str(low_idx.date()) if hasattr(low_idx, 'date') else str(low_idx)
        cup_low_pivot = Pivot(len(df) - lookback + low_pos, low_min, 'VALLEY', date_low)

        return Pattern(
            type='CUP_HANDLE',
            direction='bullish',
            pivots=[cup_low_pivot],
            neckline=neckline,
            end_idx=len(df) - 1
        )

    @staticmethod
    def detect_all(df: pd.DataFrame, pivots: List[Pivot]) -> List[Pattern]:
        patterns = []

        pat = PatternDetector.detect_double_bottom(pivots, df)
        if pat:
            patterns.append(pat)

        pat = PatternDetector.detect_double_top(pivots, df)
        if pat:
            patterns.append(pat)

        pat = PatternDetector.detect_hs_bottom(pivots, df)
        if pat:
            patterns.append(pat)

        pat = PatternDetector.detect_hs_top(pivots, df)
        if pat:
            patterns.append(pat)

        pat = PatternDetector.detect_cup_handle(df, pivots)
        if pat:
            patterns.append(pat)

        return patterns


# ============================================================
# 預判觸發機制
# ============================================================
class PredictiveTrigger:

    @staticmethod
    def check_trigger(df: pd.DataFrame, pattern: Pattern) -> bool:
        if df is None or df.empty:
            return False

        current_close = df['Close'].iloc[-1]

        if pattern.direction == 'bullish':
            right_low = PredictiveTrigger._get_right_low(pattern)
            if right_low and current_close > right_low * (1 + TRIGGER_THRESHOLD):
                return True
        else:
            right_high = PredictiveTrigger._get_right_high(pattern)
            if right_high and current_close < right_high * (1 - TRIGGER_THRESHOLD):
                return True

        return False

    @staticmethod
    def check_invalidation(df: pd.DataFrame, pattern: Pattern) -> bool:
        if df is None or df.empty:
            return False

        current_close = df['Close'].iloc[-1]

        if pattern.direction == 'bullish':
            right_low = PredictiveTrigger._get_right_low(pattern)
            if right_low and current_close < right_low:
                return True
        else:
            right_high = PredictiveTrigger._get_right_high(pattern)
            if right_high and current_close > right_high:
                return True

        return False

    @staticmethod
    def _get_right_low(pattern: Pattern) -> Optional[float]:
        if pattern.type == 'DOUBLE_BOTTOM':
            for p in pattern.pivots:
                if p.type == 'VALLEY':
                    return p.price
        elif pattern.type == 'HS_BOTTOM':
            if pattern.pivots and pattern.pivots[-1].type == 'VALLEY':
                return pattern.pivots[-1].price
        elif pattern.type == 'CUP_HANDLE':
            if pattern.pivots:
                return pattern.pivots[0].price
        return None

    @staticmethod
    def _get_right_high(pattern: Pattern) -> Optional[float]:
        if pattern.type == 'DOUBLE_TOP':
            for p in pattern.pivots:
                if p.type == 'PEAK':
                    return p.price
        elif pattern.type == 'HS_TOP':
            if pattern.pivots and pattern.pivots[-1].type == 'PEAK':
                return pattern.pivots[-1].price
        return None


# ============================================================
# 評分系統
# ============================================================
class QualityScorer:

    @staticmethod
    def score(df: pd.DataFrame, pattern: Pattern) -> Tuple[int, Dict]:
        breakdown = {'volume': 0, 'trend': 0, 'morphology': 0}

        vol_score = QualityScorer._score_volume(df, pattern)
        breakdown['volume'] = vol_score

        trend_score = QualityScorer._score_trend(df, pattern)
        breakdown['trend'] = trend_score

        morph_score = QualityScorer._score_morphology(pattern)
        breakdown['morphology'] = morph_score

        total = (
            vol_score * WEIGHT_VOLUME +
            trend_score * WEIGHT_TREND +
            morph_score * WEIGHT_MORPHOLOGY
        )

        return int(round(total)), breakdown

    @staticmethod
    def _score_volume(df: pd.DataFrame, pattern: Pattern) -> int:
        score = 0

        if len(df) < 40:
            return 50

        vol_ma20 = df['Volume'].rolling(20).mean()

        end_idx = pattern.end_idx
        if end_idx < 20 or end_idx >= len(df):
            return 50

        if pattern.type in ['DOUBLE_BOTTOM', 'HS_BOTTOM', 'CUP_HANDLE']:
            mid_idx = end_idx - 10
            if mid_idx > 10:
                left_vol = df['Volume'].iloc[mid_idx-10:mid_idx].mean()
                right_vol = df['Volume'].iloc[mid_idx:end_idx].mean()
                if right_vol < left_vol:
                    score += 50

        today_vol = df['Volume'].iloc[-1]
        avg_vol = vol_ma20.iloc[-1] if not pd.isna(vol_ma20.iloc[-1]) else df['Volume'].mean()
        if today_vol > avg_vol:
            score += 50

        return min(100, score)

    @staticmethod
    def _score_trend(df: pd.DataFrame, pattern: Pattern) -> int:
        if len(df) < 200:
            return 50

        current_close = df['Close'].iloc[-1]

        ma200 = df['Close'].rolling(200).mean().iloc[-1]

        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        std20 = df['Close'].rolling(20).std().iloc[-1]
        bb_lower = ma20 - 2 * std20
        bb_upper = ma20 + 2 * std20

        score = 0

        if pattern.direction == 'bullish':
            if current_close <= bb_lower * 1.02:
                score = 100
            elif current_close <= ma200 * 1.05:
                score = 80
            elif current_close <= ma20:
                score = 50
        else:
            if current_close >= bb_upper * 0.98:
                score = 100
            elif current_close >= ma200 * 0.95:
                score = 80
            elif current_close >= ma20:
                score = 50

        return score

    @staticmethod
    def _score_morphology(pattern: Pattern) -> int:
        score = 0

        if not pattern.pivots or len(pattern.pivots) < 2:
            return 50

        if pattern.type in ['DOUBLE_BOTTOM', 'DOUBLE_TOP'] and len(pattern.pivots) >= 3:
            p_left = pattern.pivots[0]
            p_mid = pattern.pivots[1]
            p_right = pattern.pivots[2]
            left_time = p_mid.idx - p_left.idx
            right_time = p_right.idx - p_mid.idx
            if left_time > 0:
                time_ratio = abs(left_time - right_time) / left_time
                if time_ratio < 0.2:
                    score += 50
                elif time_ratio < 0.4:
                    score += 25

        if pattern.type == 'DOUBLE_BOTTOM' and len(pattern.pivots) >= 3:
            left_price = pattern.pivots[0].price
            right_price = pattern.pivots[2].price if len(pattern.pivots) > 2 else pattern.pivots[1].price
            price_diff = abs(left_price - right_price) / left_price
            if price_diff < 0.01:
                score += 50
            elif price_diff < 0.02:
                score += 35
            elif price_diff < 0.03:
                score += 20
        elif pattern.type == 'DOUBLE_TOP' and len(pattern.pivots) >= 3:
            left_price = pattern.pivots[0].price
            right_price = pattern.pivots[2].price if len(pattern.pivots) > 2 else pattern.pivots[1].price
            price_diff = abs(left_price - right_price) / left_price
            if price_diff < 0.01:
                score += 50
            elif price_diff < 0.02:
                score += 35
            elif price_diff < 0.03:
                score += 20
        else:
            score += 30

        return min(100, score)


# ============================================================
# 目標價計算
# ============================================================
class TargetCalculator:

    @staticmethod
    def calculate(df: pd.DataFrame, pattern: Pattern) -> Dict:
        result = {
            'entry_zone': round(df['Close'].iloc[-1], 2),
            'stop_loss': 0.0,
            'target_neckline': round(pattern.neckline, 2),
            'measured_move_target': 0.0
        }

        if pattern.type == 'DOUBLE_BOTTOM':
            right_low = None
            for p in pattern.pivots:
                if p.type == 'VALLEY':
                    right_low = p.price
                    break
            if right_low:
                result['stop_loss'] = round(right_low * 0.99, 2)

            avg_bottom = sum(p.price for p in pattern.pivots if p.type == 'VALLEY') / max(1, sum(1 for p in pattern.pivots if p.type == 'VALLEY'))
            depth = pattern.neckline - avg_bottom
            result['measured_move_target'] = round(pattern.neckline + depth, 2)

        elif pattern.type == 'DOUBLE_TOP':
            right_high = None
            for p in pattern.pivots:
                if p.type == 'PEAK':
                    right_high = p.price
                    break
            if right_high:
                result['stop_loss'] = round(right_high * 1.01, 2)

            avg_top = sum(p.price for p in pattern.pivots if p.type == 'PEAK') / max(1, sum(1 for p in pattern.pivots if p.type == 'PEAK'))
            depth = avg_top - pattern.neckline
            result['measured_move_target'] = round(pattern.neckline - depth, 2)

        elif pattern.type == 'HS_BOTTOM':
            head_low = min(p.price for p in pattern.pivots if p.type == 'VALLEY')
            result['stop_loss'] = round(head_low * 0.99, 2)

            depth = pattern.neckline - head_low
            result['measured_move_target'] = round(pattern.neckline + depth, 2)

        elif pattern.type == 'HS_TOP':
            head_high = max(p.price for p in pattern.pivots if p.type == 'PEAK')
            result['stop_loss'] = round(head_high * 1.01, 2)

            depth = head_high - pattern.neckline
            result['measured_move_target'] = round(pattern.neckline - depth, 2)

        elif pattern.type == 'CUP_HANDLE':
            if pattern.pivots:
                result['stop_loss'] = round(pattern.pivots[0].price * 0.99, 2)

            cup_depth = pattern.neckline - pattern.pivots[0].price if pattern.pivots else 0
            result['measured_move_target'] = round(pattern.neckline + cup_depth, 2)

        return result


# ============================================================
# 市場掃描器
# ============================================================
class MarketScanner:
    def __init__(self, dm: DataManager):
        self.dm = dm
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def scan(self, market: str, tickers: List[str], show_progress: bool = True) -> List[PatternSignal]:
        results = []
        ok, fail = 0, 0
        n = len(tickers)

        console.print(f'\n[bold cyan]掃描 {market} ({n} 檔 x {len(TIMEFRAMES)} 時框)[/bold cyan]')

        iterator = tqdm(tickers, desc=f'{market}', unit='檔') if show_progress else tickers

        for ticker in iterator:
            if self.stop_flag:
                console.print(f'\n[yellow]使用者中止 (已完成 {ok+fail}/{n})[/yellow]')
                break

            try:
                df_daily = self.dm.get(ticker, market)
                if df_daily is None or len(df_daily) < 60:
                    fail += 1
                    continue

                df_daily = preprocess(df_daily)
                if df_daily is None or len(df_daily) < 60:
                    fail += 1
                    continue

                for tf in TIMEFRAMES:
                    df = DataManager.resample(df_daily, tf)
                    if df is None or len(df) < 60:
                        continue

                    pivots = zigzag(df)
                    if len(pivots) < 4:
                        continue

                    patterns = PatternDetector.detect_all(df, pivots)

                    for pat in patterns:
                        if not PredictiveTrigger.check_trigger(df, pat):
                            continue

                        if PredictiveTrigger.check_invalidation(df, pat):
                            continue

                        score, breakdown = QualityScorer.score(df, pat)
                        if score < QUALITY_THRESHOLD:
                            continue

                        targets = TargetCalculator.calculate(df, pat)

                        entry = targets['entry_zone']
                        stop = targets['stop_loss']
                        target1 = targets['target_neckline']

                        if pat.direction == 'bullish' and entry > stop:
                            risk = entry - stop
                            reward = target1 - entry
                            rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                        elif pat.direction == 'bearish' and stop > entry:
                            risk = stop - entry
                            reward = entry - target1
                            rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                        else:
                            rr_ratio = 0

                        signal = PatternSignal(
                            symbol=f"{ticker}.{market}",
                            pattern_type=pat.type,
                            signal_type='PREDICTIVE_LONG' if pat.direction == 'bullish' else 'PREDICTIVE_SHORT',
                            timestamp=str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                            timeframe=tf,
                            quality_score=score,
                            entry_zone=targets['entry_zone'],
                            stop_loss=targets['stop_loss'],
                            target_neckline=targets['target_neckline'],
                            measured_move_target=targets['measured_move_target'],
                            risk_reward_ratio=rr_ratio,
                            pattern_details={
                                'pivots': [(p.date, p.price, p.type) for p in pat.pivots],
                                'neckline': pat.neckline,
                            },
                            score_breakdown=breakdown
                        )
                        results.append(signal)

                ok += 1
            except Exception:
                fail += 1

            if show_progress:
                iterator.set_postfix({
                    'OK': ok, 'Fail': fail,
                    '訊號': len(results),
                    '快取': self.dm.stats['cache']
                })

        s = self.dm.stats
        yf_info = f' yf={s["yf_fallback"]}' if s['yf_fallback'] else ''
        console.print(f'完成 {ok}/{ok+fail} | API: US={s["api_us"]} TW={s["api_tw"]}{yf_info} | 快取={s["cache"]} | 錯誤={s["err"]}')
        console.print(f'[green]找到 {len(results)} 個預判訊號 (品質>={QUALITY_THRESHOLD})[/green]')
        return results

    @staticmethod
    def to_df(results: List[PatternSignal]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame(columns=['代號','市場','週期','形態','訊號','分數','進場','停損','目標1','目標2','R:R'])

        pattern_names = {
            'DOUBLE_BOTTOM': 'W底',
            'DOUBLE_TOP': 'M頭',
            'HS_BOTTOM': '頭肩底',
            'HS_TOP': '頭肩頂',
            'CUP_HANDLE': '杯柄'
        }

        rows = []
        for r in results:
            symbol_parts = r.symbol.split('.')
            ticker = symbol_parts[0]
            market = symbol_parts[1] if len(symbol_parts) > 1 else ''

            rows.append({
                '代號': ticker,
                '市場': market,
                '週期': TF_LABELS.get(r.timeframe, r.timeframe),
                '形態': pattern_names.get(r.pattern_type, r.pattern_type),
                '訊號': '做多' if 'LONG' in r.signal_type else '做空',
                '分數': r.quality_score,
                '進場': r.entry_zone,
                '停損': r.stop_loss,
                '目標1': r.target_neckline,
                '目標2': r.measured_move_target,
                'R:R': f"1:{r.risk_reward_ratio}"
            })

        return pd.DataFrame(rows).sort_values('分數', ascending=False).reset_index(drop=True)


# ============================================================
# 結果儲存
# ============================================================
def save_results(results: List[PatternSignal], market: str, mode: str) -> Dict[str, str]:
    if not results:
        return {}

    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H%M')
    base_name = f"scan_{market}_{date_str}_{time_str}"

    csv_path = RESULTS_DIR / f"{base_name}.csv"
    json_path = RESULTS_DIR / f"{base_name}.json"

    df = MarketScanner.to_df(results)
    df.insert(0, '掃描時間', datetime.now().strftime('%Y-%m-%d %H:%M'))
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    json_data = {
        'scan_info': {
            'timestamp': datetime.now().isoformat(),
            'market': market,
            'mode': mode,
            'quality_threshold': QUALITY_THRESHOLD,
            'total_signals': len(results),
        },
        'signals': []
    }

    for r in results:
        json_data['signals'].append({
            'symbol': r.symbol,
            'pattern_type': r.pattern_type,
            'signal_type': r.signal_type,
            'timestamp': r.timestamp,
            'timeframe': r.timeframe,
            'quality_score': r.quality_score,
            'entry_zone': r.entry_zone,
            'stop_loss': r.stop_loss,
            'target_neckline': r.target_neckline,
            'measured_move_target': r.measured_move_target,
            'risk_reward_ratio': r.risk_reward_ratio,
            'score_breakdown': r.score_breakdown,
            'pattern_details': r.pattern_details,
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return {'csv': str(csv_path), 'json': str(json_path)}


def list_results() -> List[Dict]:
    files = list(RESULTS_DIR.glob('scan_*.csv'))
    if not files:
        return []

    rows = []
    for f in sorted(files, reverse=True):
        parts = f.stem.split('_')
        market = parts[1] if len(parts) > 1 else '?'
        date = parts[2] if len(parts) > 2 else '?'
        time_part = parts[3] if len(parts) > 3 else ''

        rows.append({
            'file': f.name,
            'date': f"{date} {time_part[:2]}:{time_part[2:]}" if time_part else date,
            'market': market,
            'size': f"{f.stat().st_size / 1024:.1f} KB"
        })

    return rows


# ============================================================
# 圖表繪製
# ============================================================
def plot_chart(dm: DataManager, signal: PatternSignal, output_file: str = None):
    ticker = signal.symbol.split('.')[0]
    market = signal.symbol.split('.')[1] if '.' in signal.symbol else 'TW'

    df = dm.get(ticker, market)
    if df is None:
        console.print(f'[red]無法取得 {ticker} 數據[/red]')
        return

    df = preprocess(df)
    df = DataManager.resample(df, signal.timeframe)
    tf_label = TF_LABELS.get(signal.timeframe, signal.timeframe)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='K線',
        increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)

    colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], marker_color=colors,
        name='成交量', opacity=0.6
    ), row=2, col=1)

    fig.add_hline(y=signal.entry_zone, line_dash="solid", line_color="blue",
                  annotation_text=f"進場 {signal.entry_zone}", row=1, col=1)
    fig.add_hline(y=signal.stop_loss, line_dash="dash", line_color="red",
                  annotation_text=f"停損 {signal.stop_loss}", row=1, col=1)
    fig.add_hline(y=signal.target_neckline, line_dash="dot", line_color="green",
                  annotation_text=f"目標1 {signal.target_neckline}", row=1, col=1)
    fig.add_hline(y=signal.measured_move_target, line_dash="dot", line_color="darkgreen",
                  annotation_text=f"目標2 {signal.measured_move_target}", row=1, col=1)

    pattern_names = {
        'DOUBLE_BOTTOM': 'W底', 'DOUBLE_TOP': 'M頭',
        'HS_BOTTOM': '頭肩底', 'HS_TOP': '頭肩頂', 'CUP_HANDLE': '杯柄'
    }
    clr = 'green' if 'LONG' in signal.signal_type else 'red'
    fig.add_annotation(
        x=df.index[-1], y=df['High'].max(),
        text=f"<b>{pattern_names.get(signal.pattern_type, signal.pattern_type)}</b><br>"
             f"品質: {signal.quality_score} | R:R 1:{signal.risk_reward_ratio}",
        showarrow=True, font=dict(color=clr, size=14),
        bgcolor='white', bordercolor=clr, borderwidth=2
    )

    fig.update_layout(
        title=f'{ticker} ({market}) - {tf_label}',
        xaxis_rangeslider_visible=False,
        height=600, showlegend=False, template='plotly_white'
    )

    if output_file:
        fig.write_html(output_file)
        console.print(f'[green]圖表已儲存: {output_file}[/green]')
    else:
        fig.show()


# ============================================================
# 顯示結果表格
# ============================================================
def display_results_table(results: List[PatternSignal]):
    if not results:
        console.print('[yellow]未找到符合條件的訊號[/yellow]')
        return

    pattern_names = {
        'DOUBLE_BOTTOM': 'W底', 'DOUBLE_TOP': 'M頭',
        'HS_BOTTOM': '頭肩底', 'HS_TOP': '頭肩頂', 'CUP_HANDLE': '杯柄'
    }

    table = Table(title=f"掃描結果 ({len(results)} 個訊號)", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("代號", style="bold")
    table.add_column("市場")
    table.add_column("週期")
    table.add_column("形態")
    table.add_column("訊號")
    table.add_column("分數", justify="right")
    table.add_column("進場", justify="right")
    table.add_column("停損", justify="right")
    table.add_column("目標1", justify="right")
    table.add_column("R:R", justify="right")

    sorted_results = sorted(results, key=lambda x: x.quality_score, reverse=True)

    for i, r in enumerate(sorted_results, 1):
        symbol_parts = r.symbol.split('.')
        ticker = symbol_parts[0]
        market = symbol_parts[1] if len(symbol_parts) > 1 else ''
        signal_style = "green" if 'LONG' in r.signal_type else "red"

        table.add_row(
            str(i),
            ticker,
            market,
            TF_LABELS.get(r.timeframe, r.timeframe),
            pattern_names.get(r.pattern_type, r.pattern_type),
            f"[{signal_style}]{'做多' if 'LONG' in r.signal_type else '做空'}[/{signal_style}]",
            str(r.quality_score),
            str(r.entry_zone),
            str(r.stop_loss),
            str(r.target_neckline),
            f"1:{r.risk_reward_ratio}"
        )

    console.print(table)


# ============================================================
# 互動式選單
# ============================================================
def interactive_mode():
    console.print(Panel.fit(
        "[bold cyan]智慧型 K 線形態辨識系統 v2.0[/bold cyan]\n"
        "[dim]預判型系統 - 在形態完成前提前識別潛在機會[/dim]\n\n"
        "支援形態: W底, M頭, 頭肩底, 頭肩頂, 杯柄\n"
        f"品質門檻: {QUALITY_THRESHOLD} 分 | 時框: 日線, 週線, 月線",
        title="歡迎使用"
    ))

    fm_token, tg_key = Config.get_api_keys()

    while True:
        console.print("\n[bold]請選擇操作:[/bold]")
        console.print("  1. 快速掃描 (精選 ~120 檔)")
        console.print("  2. 全市場掃描 (需要 API Key)")
        console.print("  3. 設定 API Key")
        console.print("  4. 查看歷史結果")
        console.print("  5. 離開")

        choice = Prompt.ask("選擇", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "5":
            console.print("[cyan]再見![/cyan]")
            break

        elif choice == "3":
            console.print("\n[bold]設定 API Key[/bold]")
            console.print("[dim]留空則保持原設定[/dim]")
            new_fm = Prompt.ask("FinMind Token (台股)", default=fm_token or "")
            new_tg = Prompt.ask("Tiingo API Key (美股)", default=tg_key or "")
            Config.set_api_keys(new_fm, new_tg)
            fm_token, tg_key = new_fm or fm_token, new_tg or tg_key
            console.print("[green]設定已儲存![/green]")

        elif choice == "4":
            files = list_results()
            if not files:
                console.print("[yellow]尚無儲存的掃描結果[/yellow]")
            else:
                table = Table(title="歷史掃描結果")
                table.add_column("檔案")
                table.add_column("日期")
                table.add_column("市場")
                table.add_column("大小")
                for f in files[:20]:
                    table.add_row(f['file'], f['date'], f['market'], f['size'])
                console.print(table)
                console.print(f"[dim]共 {len(files)} 個檔案，存放於 results/ 目錄[/dim]")

        else:
            mode = 'quick' if choice == "1" else 'full'

            console.print("\n[bold]選擇市場:[/bold]")
            console.print("  1. 美股 (US)")
            console.print("  2. 台股 (TW)")
            console.print("  3. 全部 (US + TW)")
            market_choice = Prompt.ask("選擇", choices=["1", "2", "3"], default="1")
            market_map = {"1": "US", "2": "TW", "3": "ALL"}
            market = market_map[market_choice]

            run_scan(market, mode, fm_token, tg_key)


def run_scan(market: str, mode: str, fm_token: str = '', tg_key: str = ''):
    dm = DataManager(fm_token, tg_key)
    scanner = MarketScanner(dm)

    results = []

    try:
        if mode == 'quick':
            tickers_us = QUICK_LIST['US'] if market in ('US', 'ALL') else []
            tickers_tw = QUICK_LIST['TW'] if market in ('TW', 'ALL') else []
        else:
            tickers_us = TickerRegistry.fetch_us(tg_key) if market in ('US', 'ALL') else []
            tickers_tw = TickerRegistry.fetch_tw(fm_token) if market in ('TW', 'ALL') else []

        n_us, n_tw = len(tickers_us), len(tickers_tw)
        console.print(f'\n美股: {n_us} 檔 | 台股: {n_tw} 檔 | 合計: {n_us + n_tw} 檔')

        if tickers_us:
            results.extend(scanner.scan('US', tickers_us))
        if tickers_tw:
            results.extend(scanner.scan('TW', tickers_tw))

        if results:
            display_results_table(results)

            saved = save_results(results, market, mode)
            if saved:
                console.print(f"\n[green]結果已儲存:[/green]")
                console.print(f"  CSV:  {saved['csv']}")
                console.print(f"  JSON: {saved['json']}")

            # 簡化的圖表瀏覽模式
            sorted_results = sorted(results, key=lambda x: x.quality_score, reverse=True)
            pattern_names = {'DOUBLE_BOTTOM': 'W底', 'DOUBLE_TOP': 'M頭',
                           'HS_BOTTOM': '頭肩底', 'HS_TOP': '頭肩頂', 'CUP_HANDLE': '杯柄'}

            console.print(f"\n[bold cyan]輸入編號查看圖表 (q 離開):[/bold cyan]")

            while True:
                idx = Prompt.ask("編號", default="q")
                if idx.lower() == 'q' or idx == '':
                    break
                if idx.isdigit():
                    num = int(idx)
                    if 1 <= num <= len(sorted_results):
                        plot_chart(dm, sorted_results[num - 1])
                    else:
                        console.print(f'[yellow]請輸入 1-{len(sorted_results)} 之間的數字[/yellow]')
                else:
                    console.print('[yellow]請輸入數字或 q 離開[/yellow]')
        else:
            console.print('[yellow]未找到符合條件的訊號[/yellow]')

    except Exception as e:
        console.print(f'[red]錯誤: {e}[/red]')
        import traceback
        traceback.print_exc()


# ============================================================
# 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='智慧型 K 線形態辨識系統 v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  uv run python main.py                    # 互動模式
  uv run python main.py --market US        # 快速掃描美股
  uv run python main.py --market TW --mode full  # 全市場掃描台股
  uv run python main.py --list             # 查看歷史結果
        """
    )
    parser.add_argument('--market', '-m', choices=['US', 'TW', 'ALL'],
                        help='市場 (US/TW/ALL)')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                        help='掃描模式 (quick=精選, full=全市場)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出歷史掃描結果')
    parser.add_argument('--finmind-token', help='FinMind API Token')
    parser.add_argument('--tiingo-key', help='Tiingo API Key')

    args = parser.parse_args()

    if args.list:
        files = list_results()
        if not files:
            console.print('[yellow]尚無儲存的掃描結果[/yellow]')
        else:
            table = Table(title="歷史掃描結果")
            table.add_column("檔案")
            table.add_column("日期")
            table.add_column("市場")
            table.add_column("大小")
            for f in files:
                table.add_row(f['file'], f['date'], f['market'], f['size'])
            console.print(table)
        return

    if args.finmind_token or args.tiingo_key:
        Config.set_api_keys(args.finmind_token or '', args.tiingo_key or '')

    fm_token, tg_key = Config.get_api_keys()

    if args.market:
        run_scan(args.market, args.mode, fm_token, tg_key)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
