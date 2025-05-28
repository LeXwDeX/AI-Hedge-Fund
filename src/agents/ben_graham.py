from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import math

class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def ben_graham_agent(state: AgentState):
    """
    使用AKSHARE原生行情数据字段进行格雷厄姆风格分析。
    仅用行情数据（如收盘价、成交量等），所有分析逻辑直接用AKSHARE字段。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "获取AKSHARE行情数据")
        # 获取最近10天行情数据
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "score": 0, "max_score": 3, "details": "无行情数据"}
            continue
        # 只取最近10条
        prices = prices[-10:]

        progress.update_status("ben_graham_agent", ticker, "分析收盘价稳定性")
        earnings_analysis = analyze_earnings_stability_akshare(prices)

        progress.update_status("ben_graham_agent", ticker, "分析成交量与换手率")
        strength_analysis = analyze_financial_strength_akshare(prices)

        progress.update_status("ben_graham_agent", ticker, "分析估值（收盘价/最高/最低）")
        valuation_analysis = analyze_valuation_graham_akshare(prices)

        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 3  # 降级后每项1分

        if total_score == max_possible_score:
            signal = "bullish"
        elif total_score == 0:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "earnings_analysis": earnings_analysis,
            "strength_analysis": strength_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status("ben_graham_agent", ticker, "生成格雷厄姆风格分析")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {
            "signal": graham_output.signal,
            "confidence": graham_output.confidence,
            "reasoning": graham_output.reasoning,
        }

        progress.update_status("ben_graham_agent", ticker, "Done", analysis=graham_output.reasoning)

    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    progress.update_status("ben_graham_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}

# 新版分析函数：全部用 AKSHARE 字段
def analyze_earnings_stability_akshare(prices: list[dict]) -> dict:
    """
    用收盘价判断“盈利稳定性”：10日内收盘价波动小于5%记1分，否则0分。
    """
    if not prices or len(prices) < 2:
        return {"score": 0, "details": "无足够收盘价数据"}
    closes = [float(p["收盘"]) for p in prices if "收盘" in p and p["收盘"] is not None]
    if not closes or len(closes) < 2:
        return {"score": 0, "details": "收盘价数据不足"}
    min_close, max_close = min(closes), max(closes)
    if min_close == 0:
        return {"score": 0, "details": "收盘价为0"}
    fluct = (max_close - min_close) / min_close
    if fluct < 0.05:
        return {"score": 1, "details": "10日收盘价波动小于5%，盈利稳定"}
    else:
        return {"score": 0, "details": "10日收盘价波动较大"}

def analyze_financial_strength_akshare(prices: list[dict]) -> dict:
    """
    用换手率均值判断“财务稳健”：10日平均换手率<10%记1分，否则0分。
    """
    if not prices:
        return {"score": 0, "details": "无行情数据"}
    turnover = [float(p["换手率"]) for p in prices if "换手率" in p and p["换手率"] not in (None, "")]
    if not turnover:
        return {"score": 0, "details": "无换手率数据"}
    avg_turnover = sum(turnover) / len(turnover)
    if avg_turnover < 10:
        return {"score": 1, "details": f"10日平均换手率{avg_turnover:.2f}%，流动性稳健"}
    else:
        return {"score": 0, "details": f"10日平均换手率{avg_turnover:.2f}%，流动性偏高"}

def analyze_valuation_graham_akshare(prices: list[dict]) -> dict:
    """
    用最高/最低/收盘价判断“估值”：收盘价接近最低价记1分，否则0分。
    """
    if not prices:
        return {"score": 0, "details": "无行情数据"}
    closes = [float(p["收盘"]) for p in prices if "收盘" in p and p["收盘"] is not None]
    lows = [float(p["最低"]) for p in prices if "最低" in p and p["最低"] is not None]
    if not closes or not lows:
        return {"score": 0, "details": "收盘价或最低价数据不足"}
    close = closes[-1]
    low = lows[-1]
    if low == 0:
        return {"score": 0, "details": "最低价为0"}
    if (close - low) / low < 0.03:
        return {"score": 1, "details": "最新收盘价接近最低价，估值较低"}
    else:
        return {"score": 0, "details": "收盘价高于最低价，估值一般"}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是本杰明·格雷厄姆风格的AI投资分析师，请严格依据格雷厄姆的价值投资原则给出投资信号：
1. 坚持安全边际，只在价格低于内在价值时买入（如使用格雷厄姆数、净流动资产法等）
2. 强调公司财务稳健（低杠杆、充足流动资产）
3. 偏好多年稳定盈利的企业
4. 有分红记录更佳，增加安全性
5. 避免投机或高增长假设，聚焦已验证的财务指标

推理时请做到：
1. 详细说明影响你决策的关键估值指标（如格雷厄姆数、NCAV、市盈率等）
2. 强调具体的财务稳健性指标（流动比率、负债水平等）
3. 明确引用盈利的稳定性或波动性
4. 给出定量证据，数字要精确
5. 将当前指标与格雷厄姆的标准进行对比（如“流动比率2.5高于格雷厄姆最低2.0”）
6. 用格雷厄姆保守、理性的分析风格表达

例如，看多时：“该股以35%的折价交易于净流动资产值，安全边际充足。流动比率2.5、负债率0.3显示财务状况稳健……”
例如，看空时：“尽管盈利稳定，但当前股价50美元高于我们计算的格雷厄姆数35美元，缺乏安全边际。此外，流动比率仅1.2，低于格雷厄姆偏好的2.0标准……”

请返回理性建议：bullish、bearish或neutral，并给出0-100的置信度和详细推理。
""",
            ),
            (
                "human",
                """请根据以下分析，生成一份格雷厄姆风格的投资信号：

{ticker}的分析数据：
{analysis_data}

请严格按如下JSON格式返回：
{{
  "signal": "bullish" 或 "bearish" 或 "neutral",
  "confidence": 0-100之间的浮点数,
  "reasoning": "string"
}}
""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )
