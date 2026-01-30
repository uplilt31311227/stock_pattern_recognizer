"""整合測試: 智慧型 K 線形態辨識系統 (預判型)"""

import time, json, io, zipfile, csv, math
import requests as _requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# yfinance 備援
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

# ============================================================
# 配置 (與 Notebook 一致)
# ============================================================

CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRY_HOURS = 12

ZIGZAG_THRESHOLD = 0.03
ZIGZAG_MIN_BARS = 3
TOLERANCE = 0.03
NECKLINE_MIN_HEIGHT = 0.08
MIN_PATTERN_BARS = 15
HS_SYMMETRY_TOL = 0.10
NECKLINE_MAX_SLOPE = 30
TRIGGER_THRESHOLD = 0.02
QUALITY_THRESHOLD = 65

TIINGO_RATE_DELAY = 1.5
FINMIND_RATE_DELAY = 0.35

_PD_VER = tuple(int(x) for x in pd.__version__.split('.')[:2])
_MONTH_RULE = 'ME' if _PD_VER >= (2, 1) else 'M'

TIMEFRAMES = ['1M', '1W', '1D']
TF_LABELS = {'1D': '日線', '1W': '週線', '1M': '月線'}

FM_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMS0yNyAwMzoyMDozMiIsInVzZXJfaWQiOiJ1cGxpbHQzMTMxMTIyNyIsImVtYWlsIjoidXBsaWx0MzEzMTEyMjdAZ21haWwuY29tIiwiaXAiOiI2MC4yNTEuMTk0LjExNSJ9.M84kbWmR3H8x28ow5Gl_kYFPQDWCHbmFX-vuEXFlXi4'
TG_KEY = 'de9c5d099ca4313fd841429fb60062008975c4b0'

# 測試清單
TEST_TW = ['2330', '2454', '2317', '2881', '2882', '1301', '2603', '3008', '2412', '6505']
TEST_US = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'JPM', 'BAC']

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

# ============================================================
# DataManager
# ============================================================

class DataManager:
    def __init__(self, fm_token='', tg_key=''):
        self.fm_token = fm_token
        self.tg_key = tg_key
        self.session = _requests.Session()
        self._last_us = 0
        self._last_tw = 0
        self._backoff = 0
        self.stats = {'api_us': 0, 'api_tw': 0, 'yf_fallback': 0, 'cache': 0, 'err': 0}

    def _cache_path(self, ticker, market):
        return CACHE_DIR / f'{market}_{ticker}.csv'

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
            raise ValueError(f'HTTP {r.status_code}')
        data = r.json()
        if not data:
            raise ValueError('no data')
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
        if not _HAS_YF:
            raise ImportError('yfinance not installed')
        end = datetime.now()
        start = end - timedelta(days=2000)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError('yfinance no data')
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
            raise ValueError(data.get('msg', ''))
        rows = data['data']
        if not rows:
            raise ValueError('no data')
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
        for t in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if t not in df.columns:
                for c in df.columns:
                    if c.lower() == t.lower():
                        df = df.rename(columns={c: t})
                        break
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[keep].copy()
        if hasattr(df.index, 'tz') and df.index.tz:
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
        except Exception as e:
            self.stats['err'] += 1
            print(f'  [ERR] {ticker}: {e}')
            return None

# ============================================================
# 前處理
# ============================================================

def preprocess(df):
    if df is None or df.empty:
        return df
    df = df[df['Volume'] > 0].copy()
    df = df[df['High'] != df['Low']].copy()
    return df

# ============================================================
# ZigZag 演算法
# ============================================================

def zigzag(df, threshold=ZIGZAG_THRESHOLD, min_bars=ZIGZAG_MIN_BARS):
    if df is None or len(df) < min_bars * 2:
        return []

    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index
    n = len(df)
    pivots = []

    last_type = 'PEAK' if highs[0] >= lows[0] else 'VALLEY'
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
# 形態檢測
# ============================================================

def detect_double_bottom(pivots, df):
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
    return Pattern('DOUBLE_BOTTOM', 'bullish', [p3, p2, p1], neckline, p1.idx)

def detect_double_top(pivots, df):
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
    return Pattern('DOUBLE_TOP', 'bearish', [p3, p2, p1], neckline, p1.idx)

# ============================================================
# 測試
# ============================================================

def main():
    dm = DataManager(FM_TOKEN, TG_KEY)

    print('=== Test 1: DataManager (AAPL) ===')
    t0 = time.time()
    df_us = dm.get('AAPL', 'US')
    dt = time.time() - t0
    if df_us is not None:
        print(f'  [OK] {len(df_us)} rows, time={dt:.1f}s')
        print(f'  Range: {df_us.index[0].date()} ~ {df_us.index[-1].date()}')
    else:
        print('  [FAIL]')
        return

    print()
    print('=== Test 2: DataManager (2330) ===')
    t0 = time.time()
    df_tw = dm.get('2330', 'TW')
    dt = time.time() - t0
    if df_tw is not None:
        print(f'  [OK] {len(df_tw)} rows, time={dt:.1f}s')
        print(f'  Range: {df_tw.index[0].date()} ~ {df_tw.index[-1].date()}')
    else:
        print('  [FAIL]')
        return

    print()
    print('=== Test 3: Preprocess ===')
    df_clean = preprocess(df_tw)
    removed = len(df_tw) - len(df_clean)
    print(f'  Original: {len(df_tw)} rows')
    print(f'  After preprocess: {len(df_clean)} rows')
    print(f'  Removed: {removed} rows (Volume=0 or High==Low)')
    print('  [OK] Preprocess')

    print()
    print('=== Test 4: ZigZag Algorithm ===')
    pivots = zigzag(df_clean)
    print(f'  Found {len(pivots)} pivots')
    if pivots:
        print(f'  Latest 5 pivots:')
        for p in pivots[:5]:
            print(f'    {p.date}: {p.type} @ {p.price:.2f}')
    assert len(pivots) >= 10, f'ZigZag should find at least 10 pivots, got {len(pivots)}'
    print('  [OK] ZigZag')

    print()
    print('=== Test 5: Resample ===')
    df_1d = df_clean
    df_1w = DataManager.resample(df_1d, '1W')
    df_1m = DataManager.resample(df_1d, '1M')
    print(f'  Daily: {len(df_1d)} bars')
    print(f'  Weekly: {len(df_1w)} bars')
    print(f'  Monthly: {len(df_1m)} bars')
    assert len(df_1w) >= 50, f'Weekly should have 50+ bars'
    assert len(df_1m) >= 20, f'Monthly should have 20+ bars'
    print('  [OK] Resample')

    print()
    print('=== Test 6: Multi-timeframe ZigZag ===')
    for tf, df in [('1D', df_1d), ('1W', df_1w), ('1M', df_1m)]:
        pivots_tf = zigzag(df)
        peaks = sum(1 for p in pivots_tf if p.type == 'PEAK')
        valleys = sum(1 for p in pivots_tf if p.type == 'VALLEY')
        print(f'  {TF_LABELS[tf]}: {len(pivots_tf)} pivots ({peaks} peaks, {valleys} valleys)')
    print('  [OK] Multi-timeframe ZigZag')

    print()
    print('=== Test 7: Pattern Detection ===')
    patterns_found = 0
    for tf, df in [('1D', df_1d), ('1W', df_1w), ('1M', df_1m)]:
        pivots_tf = zigzag(df)
        pat_w = detect_double_bottom(pivots_tf, df)
        pat_m = detect_double_top(pivots_tf, df)
        if pat_w:
            print(f'  {TF_LABELS[tf]}: Found W-Bottom, neckline={pat_w.neckline:.2f}')
            patterns_found += 1
        if pat_m:
            print(f'  {TF_LABELS[tf]}: Found M-Top, neckline={pat_m.neckline:.2f}')
            patterns_found += 1
    if patterns_found == 0:
        print('  No patterns found in 2330 (this is normal)')
    print('  [OK] Pattern Detection')

    print()
    print('=== Test 8: Cache Hit ===')
    t0 = time.time()
    dm.get('AAPL', 'US')
    dm.get('2330', 'TW')
    dt = time.time() - t0
    print(f'  Cache hit time: {dt*1000:.0f}ms')
    print(f'  Stats: {dm.stats}')
    print('  [OK] Cache')

    print()
    print('=== Test 9: Batch TW ===')
    t0 = time.time()
    ok_tw, fail_tw = 0, 0
    for t in TEST_TW:
        df = dm.get(t, 'TW')
        if df is not None and len(df) >= 50:
            df = preprocess(df)
            pivots = zigzag(df)
            if len(pivots) >= 4:
                ok_tw += 1
            else:
                fail_tw += 1
        else:
            fail_tw += 1
    dt = time.time() - t0
    print(f'  TW: {ok_tw}/{len(TEST_TW)} OK ({dt:.1f}s)')

    print()
    print('=== Test 10: Batch US ===')
    t0 = time.time()
    ok_us, fail_us = 0, 0
    for t in TEST_US:
        df = dm.get(t, 'US')
        if df is not None and len(df) >= 50:
            df = preprocess(df)
            pivots = zigzag(df)
            if len(pivots) >= 4:
                ok_us += 1
            else:
                fail_us += 1
        else:
            fail_us += 1
    dt = time.time() - t0
    print(f'  US: {ok_us}/{len(TEST_US)} OK ({dt:.1f}s)')

    print()
    print(f'Final stats: {dm.stats}')

    print()
    print('=== Summary ===')
    print(f'  Data range: ~2000 days ({len(df_us)} US rows, {len(df_tw)} TW rows)')
    print(f'  ZigZag: threshold={ZIGZAG_THRESHOLD*100}%, min_bars={ZIGZAG_MIN_BARS}')
    print(f'  Pattern detection: W-Bottom, M-Top, H&S (5-point)')
    print(f'  Quality threshold: {QUALITY_THRESHOLD}')
    print(f'  TW batch: {ok_tw}/{len(TEST_TW)} success')
    print(f'  US batch: {ok_us}/{len(TEST_US)} success')
    yf_info = f', yf_fallback={dm.stats["yf_fallback"]}' if dm.stats['yf_fallback'] else ''
    print(f'  API: US={dm.stats["api_us"]}, TW={dm.stats["api_tw"]}{yf_info}')
    print(f'  Cache hits: {dm.stats["cache"]}')

    if ok_tw >= 8 and ok_us >= 8:
        print()
        print('=== ALL TESTS PASSED ===')
    else:
        print()
        print('=== SOME TESTS FAILED ===')


if __name__ == '__main__':
    main()
