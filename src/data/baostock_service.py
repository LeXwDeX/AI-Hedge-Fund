import baostock as bs
import pandas as pd
import logging
from src.data.cache import cache_data, get_cached_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def login_baostock():
    """登录Baostock系统"""
    lg = bs.login()
    if lg.error_code != '0':
        logging.error(f"Baostock登录失败: {lg.error_msg}")
        return False
    return True

def logout_baostock():
    """登出Baostock系统"""
    bs.logout()

def get_financial_data(ticker, year, quarter):
    """获取财务数据并缓存"""
    if not login_baostock():
        return pd.DataFrame()
    cache_key = f"financials_{ticker}_{year}_{quarter}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    
    # 获取股票代码（带市场前缀）
    market = "sh" if ticker.startswith("6") else "sz"
    symbol = f"{market}.{ticker}"
    
    # 获取利润表数据
    profit_data = []
    rs_profit = bs.query_profit_data(code=symbol, year=year, quarter=quarter)
    while (rs_profit.error_code == '0') and rs_profit.next():
        profit_data.append(rs_profit.get_row_data())
    profit_df = pd.DataFrame(profit_data, columns=rs_profit.fields)
    
    # 获取资产负债表
    balance_data = []
    rs_balance = bs.query_balance_data(code=symbol, year=year, quarter=quarter)
    while (rs_balance.error_code == '0') and rs_balance.next():
        balance_data.append(rs_balance.get_row_data())
    balance_df = pd.DataFrame(balance_data, columns=rs_balance.fields)
    
    # 合并数据
    merged_df = pd.concat([profit_df, balance_df], axis=1)
    
    # 添加计算字段
    if 'roeAvg' in merged_df.columns and 'netProfit' in merged_df.columns:
        merged_df['ROE'] = merged_df['roeAvg']
        merged_df['EPS'] = merged_df['epsTTM']
    
    # 缓存数据（24小时）
    cache_data(cache_key, merged_df, 86400)
    
    return merged_df

def get_available_financial_fields():
    """返回可用的财务字段"""
    return [
        'code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin',
        'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare',
        'currentRatio', 'quickRatio', 'cashRatio', 'YOYLiability',
        'liabilityToAsset', 'assetToEquity', 'ROE', 'EPS'
    ]
