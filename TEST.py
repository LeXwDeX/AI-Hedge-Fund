import akshare as ak
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def test_akshare_stock_hist(ticker):
    logging.info(f"测试 AKShare A股历史行情接口: {ticker}")
    try:
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        if not hasattr(df, "columns") or df.empty:
            logging.error(f"DataFrame 为空, ticker={ticker}")
            return
        logging.info(f"DataFrame head:\n{df.head()}")
        logging.info(f"columns: {df.columns.tolist()}")
        logging.info(f"index: {df.index}")
        if "日期" in df.columns:
            logging.info(f"日期字段类型: {type(df['日期'].iloc[0])}")
    except Exception as e:
        logging.error(f"获取A股历史行情失败: {e}, ticker={ticker}")

def test_akshare_financials(ticker):
    logging.info(f"测试 AKShare A股主要财务指标接口: {ticker}")
    try:
        # 先尝试带后缀
        for symbol in [ticker, f"{ticker}.SZ", f"{ticker}.SH"]:
            try:
                print(f"尝试 symbol={symbol}")
                df = ak.stock_financial_abstract(symbol=symbol)
                print(f"symbol={symbol} 返回内容：{df}")
                if not hasattr(df, "columns") or df.empty:
                    logging.error(f"DataFrame 为空, symbol={symbol}")
                    continue
                logging.info(f"symbol={symbol} 返回类型: {type(df)}")
                logging.info(f"字段: {df.columns.tolist()}")
                logging.info(f"index: {df.index}")
                logging.info(f"前5行数据:\n{df.head()}")
            except Exception as e2:
                logging.error(f"symbol={symbol} 获取A股财务指标失败: {e2}, ticker={ticker}")
        # 新增：尝试其他财务接口
        try:
            print(f"尝试 ak.stock_financial_report_sina(symbol={ticker})")
            df_sina = ak.stock_financial_report_sina(symbol=ticker)
            print(f"ak.stock_financial_report_sina 返回内容：{df_sina}")
        except Exception as e3:
            logging.error(f"ak.stock_financial_report_sina 获取失败: {e3}, ticker={ticker}")
        try:
            print(f"尝试 ak.stock_financial_analysis_indicator(symbol={ticker})")
            df_analysis = ak.stock_financial_analysis_indicator(symbol=ticker)
            print(f"ak.stock_financial_analysis_indicator 返回内容：{df_analysis}")
        except Exception as e4:
            logging.error(f"ak.stock_financial_analysis_indicator 获取失败: {e4}, ticker={ticker}")
    except Exception as e:
        logging.error(f"获取A股财务指标失败: {e}, ticker={ticker}")

if __name__ == "__main__":
    # 只测招商银行600036
    ticker = "600036"
    print(f"==== AKShare 全面信息抓取：{ticker} ====")
    import akshare as ak

    # 1. 历史行情
    try:
        df_hist = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        print("历史行情（前5行）：")
        print(df_hist.head())
    except Exception as e:
        print(f"历史行情获取失败: {e}")

    # 2. 财务摘要
    try:
        df_fin = ak.stock_financial_abstract(symbol=ticker)
        print("财务摘要（前5行）：")
        print(df_fin.head())
    except Exception as e:
        print(f"财务摘要获取失败: {e}")

    # 3. 市值、流通市值、实时行情
    try:
        spot_df = ak.stock_zh_a_spot_em()
        row = spot_df[spot_df["代码"] == ticker]
        print("实时行情（市值/流通市值）：")
        print(row[["代码", "名称", "总市值", "流通市值"]])
    except Exception as e:
        print(f"实时行情获取失败: {e}")

    # 4. 个股基本信息（东财）
    try:
        info_df = ak.stock_individual_info_em(symbol=ticker)
        print("个股基本信息（东财）：")
        print(info_df)
    except Exception as e:
        print(f"个股基本信息获取失败: {e}")

    # 5. 个股公司简介（雪球）
    try:
        xq_df = ak.stock_individual_basic_info_xq(symbol="SH600036")
        print("雪球公司简介：")
        print(xq_df)
    except Exception as e:
        print(f"雪球公司简介获取失败: {e}")

    # 6. 验证 get_financial_metrics 市值字段
    print("验证 get_financial_metrics 市值字段：")
    from src.tools.api import get_financial_metrics
    metrics = get_financial_metrics(ticker, "2025-05-28")
    for m in metrics:
        print(f"report_period={m.report_period}, market_cap={m.market_cap}")
    if metrics and metrics[0].market_cap:
        print("市值字段自动填充验证通过。")
    else:
        print("市值字段填充失败，请检查。")


# ==== AKShare 全面信息抓取：600036 ====
# 历史行情（前5行）：
#            日期    股票代码    开盘    收盘    最高    最低      成交量           成交额    振幅   涨跌幅   涨跌额    换手率
# 0  2002-04-09  600036 -9.10 -9.06 -9.01 -9.10  4141088  4.418822e+09 -0.91  8.67  0.86  69.02
# 1  2002-04-10  600036 -9.06 -9.08 -9.05 -9.13   679455  7.166843e+08 -0.88 -0.22 -0.02  11.32
# 2  2002-04-11  600036 -9.08 -9.10 -9.06 -9.11   227883  2.409635e+08 -0.55 -0.22 -0.02   3.80
# 3  2002-04-12  600036 -9.10 -9.09 -9.07 -9.11   212565  2.240599e+08 -0.44  0.11  0.01   3.54
# 4  2002-04-15  600036 -9.09 -9.13 -9.08 -9.14   185311  1.933069e+08 -0.66 -0.44 -0.04   3.09
# 财务摘要获取失败: Expecting value: line 1 column 1 (char 0)
# 实时行情（市值/流通市值）：
#           代码    名称           总市值          流通市值
# 2013  600036  招商银行  1.105638e+12  9.043729e+11
# 个股基本信息（东财）：
#    item                 value
# 0    最新                 43.83
# 1  股票代码                600036
# 2  股票简称                  招商银行
# 3   总股本         25219845601.0
# 4   流通股         20628944429.0
# 5   总市值  1105385832691.830078
# 6  流通市值   904166634323.069946
# 7    行业                    银行
# 8  上市时间              20020409
# 雪球公司简介：
#                             item                                              value
# 0                         org_id                                           03130097
# 1                    org_name_cn                                         招商银行股份有限公司
# 2              org_short_name_cn                                               招商银行
# 3                    org_name_en                      China Merchants Bank Co.,Ltd.
# 4              org_short_name_en                                                CMB
# 5        main_operation_business                  向客户提供各种批发及零售银行产品和服务，亦自营及代客进行资金业务。
# 6                operating_scope  　　吸收公众存款；发放短期、中期和长期贷款；办理结算；办理票据贴现；发行金融债券；代理发行、...
# 7                district_encode                                             440304
# 8            org_cn_introduction  招商银行股份有限公司的主营业务是向客户提供各种批发及零售银行产品和服务，亦自营及代客进行资金...
# 9           legal_representative                                                缪建民
# 10               general_manager                                                 王良
# 11                     secretary                                                彭家文
# 12              established_date                                       544118400000
# 13                     reg_asset                                      25219845601.0
# 14                     staff_num                                             117201
# 15                     telephone                                    86-755-83198888
# 16                      postcode                                             518040
# 17                           fax                                    86-755-83195555
# 18                         email                                   cmb@cmbchina.com
# 19                   org_website                                   www.cmbchina.com
# 20                reg_address_cn                                 广东省深圳市福田区深南大道7088号
# 21                reg_address_en                                               None
# 22             office_address_cn                                 广东省深圳市福田区深南大道7088号
# 23             office_address_en                                               None
# 24               currency_encode                                             019001
# 25                      currency                                                CNY
# 26                   listed_date                                      1018281600000
# 27               provincial_name                                                广东省
# 28             actual_controller
# 29                   classi_name                                             央企国资控股
# 30                   pre_name_cn                                               None
# 31                      chairman                                                缪建民
# 32               executives_nums                                                 31
# 33              actual_issue_vol                                       1500000000.0
# 34                   issue_price                                                7.3
# 35             actual_rc_net_amt                                      10742890000.0
# 36              pe_after_issuing                                              21.47
# 37  online_success_rate_of_issue                                             1.2479
# 38            affiliate_industry           {'ind_code': 'BK0055', 'ind_name': '银行'}
# 验证 get_financial_metrics 市值字段：
# 2025-05-28 09:34:12,959 [ERROR] [AKShare] 600036.SH 财务摘要接口异常: Expecting value: line 1 column 1 (char 0), ticker=600036
# 2025-05-28 09:34:13,047 [ERROR] [AKShare] 600036 财务摘要接口异常: Expecting value: line 1 column 1 (char 0), ticker=600036
# 市值字段填充失败，请检查。