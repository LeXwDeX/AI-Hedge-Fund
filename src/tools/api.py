import datetime
import os
import pandas as pd
import requests

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """
    使用 AKShare 获取A股日K线行情，映射为 Price 列表。
    ticker: 6位股票代码（如"000001"），自动适配深沪市。
    start_date, end_date: "YYYY-MM-DD"
    """
    import akshare as ak
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # ticker 合法性校验与补全
    if not isinstance(ticker, str) or not ticker.isdigit() or len(ticker) > 6:
        logging.error(f"[AKShare] ticker 非法: {ticker}")
        return []
    ticker = ticker.zfill(6)

    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            logging.info(f"[CACHE] 命中 {ticker}，返回{len(filtered_data)}条")
            return filtered_data

    # 判断市场后缀
    if ticker.startswith("6"):
        symbol = f"{ticker}.SH"
    else:
        symbol = f"{ticker}.SZ"

    try:
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        if not isinstance(df, pd.DataFrame) or df.empty:
            logging.error(f"[AKShare] {symbol} 行情接口返回空DataFrame，ticker={ticker}, 参数: {start_date}~{end_date}")
            return []
        logging.info(f"[AKShare] {symbol} 行情获取成功，数据量: {len(df)}")
        if "日期" not in df.columns:
            logging.error(f"[AKShare] DataFrame 字段: {df.columns.tolist()}，缺少'日期'字段, ticker={ticker}")
            return []
        # 兼容日期类型
        if not isinstance(df["日期"].iloc[0], str):
            df["日期"] = df["日期"].astype(str)
        # 过滤日期
        df = df[(df["日期"] >= start_date) & (df["日期"] <= end_date)]
        prices = []
        for _, row in df.iterrows():
            price = Price(
                open=float(row["开盘"]),
                close=float(row["收盘"]),
                high=float(row["最高"]),
                low=float(row["最低"]),
                volume=int(float(row["成交量"])),
                time=row["日期"]
            )
            prices.append(price)
        # 缓存
        _cache.set_prices(ticker, [p.model_dump() for p in prices])
        logging.info(f"[AKShare] {symbol} 映射为 Price 共{len(prices)}条")
        return prices
    except Exception as e:
        logging.error(f"[AKShare] 获取A股行情失败: {e}, ticker={ticker}, symbol={symbol}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """
    使用 AKShare 获取A股主要财务指标，映射为 FinancialMetrics 列表。
    ticker: 6位股票代码（如"000001"），自动适配深沪市。
    end_date: "YYYY-MM-DD"
    """
    import akshare as ak
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            logging.info(f"[CACHE] 命中 {ticker} 财务，返回{len(filtered_data)}条")
            return filtered_data[:limit]

    # 判断市场后缀
    if not isinstance(ticker, str) or not ticker.isdigit() or len(ticker) > 6:
        logging.error(f"[AKShare] ticker 非法: {ticker}")
        return []
    ticker = ticker.zfill(6)
    if ticker.startswith("6"):
        symbol = f"{ticker}.SH"
    else:
        symbol = f"{ticker}.SZ"

    try:
        # 优先尝试带后缀
        df = None
        try:
            df = ak.stock_financial_abstract(symbol=symbol)
            if not isinstance(df, pd.DataFrame) or df.empty:
                # 如果带后缀失败，再尝试6位代码
                df = ak.stock_financial_abstract(symbol=ticker)
        except Exception as e:
            logging.error(f"[AKShare] {symbol} 财务摘要接口异常: {e}, ticker={ticker}")
            try:
                df = ak.stock_financial_abstract(symbol=ticker)
            except Exception as e2:
                logging.error(f"[AKShare] {ticker} 财务摘要接口异常: {e2}, ticker={ticker}")
                return []
        if not isinstance(df, pd.DataFrame) or df.empty:
            logging.error(f"[AKShare] {symbol}/{ticker} 财务摘要无数据，返回空DataFrame, ticker={ticker}")
            return []
        logging.info(f"[AKShare] {symbol} 财务摘要获取成功，数据量: {len(df)}")
        # 以 end_date 为最近报告期，向前取 limit 个报告期
        periods = [col for col in df.columns if col not in ["选项", "指标"] and col <= end_date]
        periods = sorted(periods, reverse=True)[:limit]
        metrics_list = []
        # 指标名与 FinancialMetrics 字段映射表（可扩展）
        indicator_map = {
            "归母净利润": "net_income",
            "营业总收入": "revenue",
            "净利润": "net_margin",
            "总资产": "total_assets",
            "总负债": "total_liabilities",
            "每股收益": "earnings_per_share",
            "每股净资产": "book_value_per_share",
            "毛利率": "gross_margin",
            "营业利润率": "operating_margin",
            "资产负债率": "debt_to_assets",
            "流动比率": "current_ratio",
            "速动比率": "quick_ratio",
            "ROE": "return_on_equity",
            "ROA": "return_on_assets",
            "每股经营现金流": "free_cash_flow_per_share",
            # ...可根据实际需求补充
        }
        # 获取所有 FinancialMetrics 字段名
        from src.data.models import FinancialMetrics as FMCls
        fm_fields = FMCls.model_fields.keys()
        # 获取市值快照
        try:
            spot_df = ak.stock_zh_a_spot_em()
        except Exception as e:
            spot_df = None
            logging.warning(f"[AKShare] 获取市值快照失败: {e}")

        for period in periods:
            # 先全部置为 None
            metric = {k: None for k in fm_fields}
            # 必填基础字段
            metric.update({
                "ticker": ticker,
                "report_period": period,
                "period": "q" if period[-4:] in ["0331", "0630", "0930"] else "y",
                "currency": "CNY",
            })
            for idx, row in df.iterrows():
                zh_name = row["指标"]
                if zh_name in indicator_map:
                    field = indicator_map[zh_name]
                    value = row.get(period, None)
                    try:
                        value = float(value) if value is not None and value != "" else None
                    except Exception:
                        value = None
                    metric[field] = value
            # 补充市值字段
            market_cap = None
            if spot_df is not None:
                try:
                    row_spot = spot_df[spot_df["代码"] == ticker]
                    if not row_spot.empty:
                        market_cap = float(row_spot.iloc[0]["总市值"])
                except Exception as e:
                    logging.warning(f"[AKShare] 市值查找失败: {e}")
            metric["market_cap"] = market_cap
            metrics_list.append(FinancialMetrics(**metric))
        # 缓存
        _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
        logging.info(f"[AKShare] {symbol} 映射为 FinancialMetrics 共{len(metrics_list)}条")
        return metrics_list
    except Exception as e:
        logging.error(f"[AKShare] 获取A股财务指标失败: {e}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """
    A股环境下暂不支持search_line_items功能，如需特定财务行项目请直接用get_financial_metrics并在本地筛选。
    若数据源不支持，始终返回空列表。
    """
    import logging
    logging.warning("[AKShare] search_line_items 暂不支持A股接口，请用 get_financial_metrics 替代。")
    return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """
    A股环境下暂不支持insider_trades功能，AKShare暂无A股高管/董监高持股变动接口。
    若数据源不支持，始终返回空列表。
    """
    import logging
    logging.warning("[AKShare] get_insider_trades 暂不支持A股接口。")
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """
    A股环境下暂不支持公司新闻聚合接口，AKShare暂无A股公司新闻API。
    若数据源不支持，始终返回空列表。
    """
    import logging
    logging.warning("[AKShare] get_company_news 暂不支持A股接口。")
    return []


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """
    A股市值可通过 get_financial_metrics 返回的 market_cap 字段获得。
    """
    import logging
    metrics = get_financial_metrics(ticker, end_date)
    if not metrics:
        logging.warning(f"[AKShare] {ticker} 未获取到市值数据")
        return None
    market_cap = getattr(metrics[0], "market_cap", None)
    if market_cap is None:
        logging.warning(f"[AKShare] {ticker} 市值字段为空")
    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
