from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def cathie_wood_agent(state: AgentState):
    """
    只用AKSHARE行情数据（ak.stock_zh_a_hist字段），做统一技术分析（动量/波动/均线/换手率等）。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent", ticker, "获取AKSHARE行情数据")
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "score": 0, "max_score": 3, "details": "无行情数据"}
            continue
        # 只取最近30条
        prices = prices[-30:]

        # 直接将行情数据和技术分析提示词交给LLM
        progress.update_status("cathie_wood_agent", ticker, "生成技术分析信号")
        cw_output = generate_cathie_wood_technical_output(
            ticker=ticker,
            prices=prices,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # A股不能做空，bearish时给出替代建议
        reasoning = cw_output.reasoning
        if cw_output.signal == "bearish":
            reasoning = f"{reasoning}（A股市场不能做空，建议观望或空仓，避免盲目操作。）"

        cw_analysis[ticker] = {
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "reasoning": reasoning
        }

        progress.update_status("cathie_wood_agent", ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name="cathie_wood_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "Cathie Wood Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = cw_analysis

    progress.update_status("cathie_wood_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}

def generate_cathie_wood_technical_output(
    ticker: str,
    prices: list[dict],
    model_name: str,
    model_provider: str,
) -> CathieWoodSignal:
    """
    只用AKSHARE行情数据，提示词要求LLM用技术分析方法（如趋势、动量、波动、均线、换手率等）给出bullish/bearish/neutral信号和详细理由。
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是技术分析专家。请只基于下方A股行情数据（包含日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率等），用你的技术分析知识（如趋势、动量、波动、均线、换手率、K线形态等）分析该股票当前的盘面特征，给出bullish（看多）、bearish（看空）、neutral（中性）信号，并详细说明理由。禁止参考任何财务、估值、成长等信息。
数据如下（最近30日）：
{prices}
请严格按如下JSON格式返回：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "reasoning": "string"
}}
"""
        ),
    ])

    prompt = template.invoke({
        "prices": json.dumps(prices, ensure_ascii=False, indent=2),
        "ticker": ticker
    })

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=create_default_cathie_wood_signal,
    )


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. R&D Intensity - shows innovation investment
    3. Gross Margin Trends - suggests pricing power and scalability
    4. Operating Leverage - demonstrates business model efficiency
    5. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")

        # Check absolute growth rate (most recent growth rate is at index 0)
        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level (most recent margin is at index 0)
        if gross_margins[0] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[0]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    # Normalize score to be out of 5
    max_possible_score = 12  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's commitment to innovation and potential for exponential growth.
    Analyzes multiple dimensions:
    1. R&D Investment Trends - measures commitment to innovation
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability of innovation
    4. Capital Allocation - reveals innovation-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

    # 1. R&D Investment Trends
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

        # Check R&D intensity trend (corrected for reverse chronological order)
        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF, adequate innovation funding capacity")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]

        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    max_possible_score = 15  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We can do
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [f"Calculated intrinsic value: ~{intrinsic_value:,.2f}", f"Market cap: ~{market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]

    return {"score": score, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> CathieWoodSignal:
    """
    Generates investment decisions in the style of Cathie Wood.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是凯茜·伍德（Cathie Wood）风格的AI投资分析师，请严格依据其创新成长投资理念给出投资信号：

1. 重点关注颠覆性创新和指数级成长机会，优先投资于AI、机器人、基因科技、金融科技等变革性行业。
2. 偏好具备远大愿景、强大研发能力和领导力的公司。
3. 注重长期趋势和大市场空间（TAM），愿意承受短期波动以追求长期高回报。
4. 关注公司技术壁垒、专利、创新能力及管理层执行力。
5. 估值可接受高溢价，但需有明确的成长逻辑和数据支撑。

推理时请做到：
- 明确指出公司所依托的创新技术/模式及其颠覆性。
- 强调收入加速、市场空间扩张等成长性指标。
- 分析公司在未来5年以上的变革潜力和行业影响力。
- 评估研发投入、创新管线、管理层远见等软硬实力。
- 给出成长型估值逻辑和定量假设。
- 风格乐观、前瞻、充满信念。

请返回最终建议（signal: bullish、neutral、bearish），并给出0-100置信度和详细推理说明。
"""
            ),
            (
                "human",
                """请根据以下分析，生成一份凯茜·伍德风格的投资信号：

{ticker}的分析数据：
{analysis_data}

请严格按如下JSON格式返回：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float (0-100),
  "reasoning": "string"
}}
"""
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        default_factory=create_default_cathie_wood_signal,
    )


# source: https://ark-invest.com
