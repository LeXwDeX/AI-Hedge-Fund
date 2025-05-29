import rich
import baostock as bs
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_baostock_financial(ticker):
    """测试Baostock财务数据接口"""
    print(f"\n{'='*50}")
    print(f"测试股票: {ticker}")
    print(f"{'='*50}")
    
    try:
        # 登录系统
        lg = bs.login()
        if lg.error_code != '0':
            print(f"登录失败: {lg.error_msg}")
            return
        
        # 获取股票代码（带市场前缀）
        market = "sh" if ticker.startswith("6") else "sz"
        symbol = f"{market}.{ticker}"
        
        # 查询最新财务数据（利润表）
        print(f"\n查询利润表数据: {symbol}")
        profit_list = []
        rs_profit = bs.query_profit_data(code=symbol, year=2024, quarter=1)
        while (rs_profit.error_code == '0') and rs_profit.next():
            profit_list.append(rs_profit.get_row_data())
        profit_df = pd.DataFrame(profit_list, columns=rs_profit.fields)
        
        # 查询资产负债表
        print(f"\n查询资产负债表: {symbol}")
        balance_list = []
        rs_balance = bs.query_balance_data(code=symbol, year=2024, quarter=1)
        while (rs_balance.error_code == '0') and rs_balance.next():
            balance_list.append(rs_balance.get_row_data())
        balance_df = pd.DataFrame(balance_list, columns=rs_balance.fields)
        
        # 显示结果
        print("\n利润表数据:")
        print(profit_df.head())
        
        print("\n资产负债表数据:")
        print(balance_df.head())
        
        # 检查关键财务字段
        key_fields = ['netProfit', 'totalAssets', 'eps', 'ROE']
        print("\n关键财务指标检查:")
        for field in key_fields:
            if field in profit_df.columns or field in balance_df.columns:
                print(f"✅ {field} 存在")
            else:
                print(f"❌ {field} 缺失")
        
        # 登出系统
        bs.logout()
        
        print("="*50)
        print("结论: 接口可用 ✅")
        print("="*50)
        
    except Exception as e:
        logging.error(f"接口调用失败: {str(e)}")
        print("="*50)
        print("结论: 接口不可用 ❌")
        print("="*50)

def display_result(df, method):
    """展示处理结果"""
    if not isinstance(df, pd.DataFrame):
        print(f"{method} 返回非DataFrame类型: {type(df)}")
        return
        
    print(f"\n{method} 返回结果:")
    print(f"数据类型: {type(df)}")
    print(f"数据形状: {df.shape}")
    
    if df.empty:
        print("警告: 返回空DataFrame")
        return
        
    print("\n前5行数据:")
    print(df.head())
    
    print("\n字段列表:")
    print(df.columns.tolist())
    
    # 检查关键财务字段
    key_fields = ['归母净利润', '营业总收入', '每股收益', '资产负债率']
    missing = [f for f in key_fields if f not in df.columns]
    
    if missing:
        print(f"\n警告: 缺少关键字段 {missing}")
    else:
        print("\n✅ 包含所有关键财务字段")

if __name__ == "__main__":
    # 测试股票600699
    test_baostock_financial("600699")
    
    # 额外测试其他股票
    print("\n\n额外测试其他股票...")
    test_baostock_financial("000001")  # 平安银行
    test_baostock_financial("600519")  # 贵州茅台
     # poetry run python src/main.py --tickers 600699 --start-date 2024-01-01 --end-date 2026-01-01                                                                                                                     