# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
"""

import csv
import logging
from datetime import datetime
from io import StringIO
from typing import Optional, List, Dict, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, is_bse_code
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource

# 可选导入本地股票映射补丁，若缺失则使用空字典兜底
try:
    from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name
except (ImportError, ModuleNotFoundError):
    STOCK_NAME_MAP = {}

    def is_meaningful_stock_name(name: str | None, stock_code: str) -> bool:
        """简单的名称有效性校验兜底"""
        if not name:
            return False
        n = str(name).strip()
        return bool(n and n.upper() != str(stock_code).strip().upper())

import os

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现

    优先级：4（最低，作为兜底）
    数据来源：Yahoo Finance

    关键策略：
    - 自动转换股票代码格式
    - 处理时区和数据格式差异
    - 失败后指数退避重试

    注意事项：
    - A 股数据可能有延迟
    - 某些股票可能无数据
    - 数据精度可能与国内源略有差异
    """

    name = "YfinanceFetcher"
    priority = int(os.getenv("YFINANCE_PRIORITY", "4"))

    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass

    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式

        Yahoo Finance 代码格式：
        - A股沪市：600519.SS (Shanghai Stock Exchange)
        - A股深市：000001.SZ (Shenzhen Stock Exchange)

        Args:
            stock_code: 原始代码，如 '600519', 'hk00700', 'AAPL'

        Returns:
            Yahoo Finance 格式代码

        Examples:
            >>> fetcher._convert_stock_code('600519')
            '600519.SS'
        """
        code = stock_code.strip().upper()

        # 已经包含后缀的情况
        if '.SS' in code or '.SZ' in code or '.HK' in code or '.BJ' in code:
            return code

        # 去除可能的 .SH 后缀
        code = code.replace('.SH', '')

        # ETF: Shanghai ETF (51xx, 52xx, 56xx, 58xx) -> .SS; Shenzhen ETF (15xx, 16xx, 18xx) -> .SZ
        if len(code) == 6:
            if code.startswith(('51', '52', '56', '58')):
                return f"{code}.SS"
            if code.startswith(('15', '16', '18')):
                return f"{code}.SZ"

        # BSE (Beijing Stock Exchange): 8xxxxx, 4xxxxx, 920xxx
        if is_bse_code(code):
            base = code.split('.')[0] if '.' in code else code
            return f"{base}.BJ"

        # A股：根据代码前缀判断市场
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据

        使用 yfinance.download() 获取历史数据

        流程：
        1. 转换股票代码格式
        2. 调用 yfinance API
        3. 处理返回数据
        """
        import yfinance as yf

        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")

        try:
            # 使用 yfinance 下载数据
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,  # 禁止进度条
                auto_adjust=True,  # 自动调整价格（复权）
                multi_level_index=True
            )

            # 筛选出 yf_code 的列, 避免多只股票数据混淆
            if isinstance(df.columns, pd.MultiIndex) and len(df.columns) > 1:
                ticker_level = df.columns.get_level_values(1)
                mask = ticker_level == yf_code
                if mask.any():
                    df = df.loc[:, mask].copy()

            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")

            return df

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据

        yfinance 返回的列名：
        Open, High, Low, Close, Volume（索引是日期）

        注意：新版 yfinance 返回 MultiIndex 列名，如 ('Close', 'AMD')
        需要先扁平化列名再进行处理

        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 处理 MultiIndex 列名（新版 yfinance 返回格式）
        # 例如: ('Close', 'AMD') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("检测到 MultiIndex 列名，进行扁平化处理")
            # 取第一级列名（Price level: Close, High, Low, etc.）
            df.columns = df.columns.get_level_values(0)

        # 重置索引，将日期从索引变为列
        df = df.reset_index()

        # 列名映射（yfinance 使用首字母大写）
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }

        df = df.rename(columns=column_mapping)

        # 计算涨跌幅（因为 yfinance 不直接提供）
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)

        # 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 平均价格
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0

        # 添加股票代码列
        df['code'] = stock_code

        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def _fetch_yf_ticker_data(self, yf, yf_code: str, name: str, return_code: str) -> Optional[Dict[str, Any]]:
        """
        通过 yfinance 拉取单个指数/股票的行情数据。

        Args:
            yf: yfinance 模块引用
            yf_code: yfinance 使用的代码（如 '000001.SS'）
            name: 指数显示名称
            return_code: 写入结果 dict 的 code 字段（如 'sh000001'）

        Returns:
            行情字典，失败时返回 None
        """
        ticker = yf.Ticker(yf_code)
        # 取近两日数据以计算涨跌幅
        hist = ticker.history(period='2d')
        if hist.empty:
            return None
        today_row = hist.iloc[-1]
        prev_row = hist.iloc[-2] if len(hist) > 1 else today_row
        price = float(today_row['Close'])
        prev_close = float(prev_row['Close'])
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        high = float(today_row['High'])
        low = float(today_row['Low'])
        # 振幅 = (最高 - 最低) / 昨收 * 100
        amplitude = ((high - low) / prev_close * 100) if prev_close else 0
        return {
            'code': return_code,
            'name': name,
            'current': price,
            'change': change,
            'change_pct': change_pct,
            'open': float(today_row['Open']),
            'high': high,
            'low': low,
            'prev_close': prev_close,
            'volume': float(today_row['Volume']),
            'amount': 0.0,  # Yahoo Finance 不提供准确成交额
            'amplitude': amplitude,
        }

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数行情 (Yahoo Finance)，支持 A 股、美股与港股。
        """
        import yfinance as yf

        # A 股指数：akshare 代码 -> (yfinance 代码, 显示名称)
        yf_mapping = {
            'sh000001': ('000001.SS', '上证指数'),
            'sz399001': ('399001.SZ', '深证成指'),
            'sz399006': ('399006.SZ', '创业板指'),
            'sh000688': ('000688.SS', '科创50'),
            'sh000016': ('000016.SS', '上证50'),
            'sh000300': ('000300.SS', '沪深300'),
        }

        results = []
        try:
            for ak_code, (yf_code, name) in yf_mapping.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_code, name, ak_code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取指数 {name} 失败: {e}")

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个 A 股指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取 A 股指数行情失败: {e}")

        return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    fetcher = YfinanceFetcher()

    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"获取失败: {e}")
