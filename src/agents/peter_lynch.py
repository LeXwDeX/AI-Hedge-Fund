from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

class PeterLynchSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def peter_lynch_agent(state: AgentState):
    """
    只用AKSHARE行情数据（ak.stock_zh_a_hist字段），做统一技术分析（动量/波动/均线/换手率等）。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    lynch_analysis = {}

    for ticker in tickers:
        progress.update_status("peter_lynch_agent", ticker, "获取AKSHARE行情数据")
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "score": 0, "max_score": 3, "details": "无行情数据"}
            continue
        # 只取最近30条
        prices = prices[-30:]

        # 直接将行情数据和技术分析提示词交给LLM
        progress.update_status("peter_lynch_agent", ticker, "生成技术分析信号")
        lynch_output = generate_lynch_technical_output(
            ticker=ticker,
            prices=prices,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # A股不能做空，bearish时给出替代建议
        reasoning = lynch_output.reasoning
        if lynch_output.signal == "bearish":
            reasoning = f"{reasoning}（A股市场不能做空，建议观望或空仓，避免盲目操作。）"

        lynch_analysis[ticker] = {
            "signal": lynch_output.signal,
            "confidence": lynch_output.confidence,
            "reasoning": reasoning
        }

        progress.update_status("peter_lynch_agent", ticker, "Done", analysis=lynch_output.reasoning)

    message = HumanMessage(content=json.dumps(lynch_analysis), name="peter_lynch_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(lynch_analysis, "Peter Lynch Agent")

    state["data"]["analyst_signals"]["peter_lynch_agent"] = lynch_analysis

    progress.update_status("peter_lynch_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}

def generate_lynch_technical_output(
    ticker: str,
    prices: list[dict],
    model_name: str,
    model_provider: str,
) -> PeterLynchSignal:
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

    def create_default_signal():
        return PeterLynchSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PeterLynchSignal,
        agent_name="peter_lynch_agent",
        default_factory=create_default_signal,
    )


def analyze_lynch_growth(financial_line_items: list) -> dict:
    """
    Evaluate growth based on revenue and EPS trends:
      - Consistent revenue growth
      - Consistent EPS growth
    Peter Lynch liked companies with steady, understandable growth,
    often searching for potential 'ten-baggers' with a long runway.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient financial data for growth analysis"}

    details = []
    raw_score = 0  # We'll sum up points, then scale to 0–10 eventually

    # 1) Revenue Growth
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2:
        latest_rev = revenues[0]
        older_rev = revenues[-1]
        if older_rev > 0:
            rev_growth = (latest_rev - older_rev) / abs(older_rev)
            if rev_growth > 0.25:
                raw_score += 3
                details.append(f"Strong revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:
                raw_score += 2
                details.append(f"Moderate revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.02:
                raw_score += 1
                details.append(f"Slight revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Flat or negative revenue growth: {rev_growth:.1%}")
        else:
            details.append("Older revenue is zero/negative; can't compute revenue growth.")
    else:
        details.append("Not enough revenue data to assess growth.")

    # 2) EPS Growth
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        if abs(older_eps) > 1e-9:
            eps_growth = (latest_eps - older_eps) / abs(older_eps)
            if eps_growth > 0.25:
                raw_score += 3
                details.append(f"Strong EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                raw_score += 2
                details.append(f"Moderate EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.02:
                raw_score += 1
                details.append(f"Slight EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative EPS growth: {eps_growth:.1%}")
        else:
            details.append("Older EPS is near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data for growth calculation.")

    # raw_score can be up to 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_lynch_fundamentals(financial_line_items: list) -> dict:
    """
    Evaluate basic fundamentals:
      - Debt/Equity
      - Operating margin (or gross margin)
      - Positive Free Cash Flow
    Lynch avoided heavily indebted or complicated businesses.
    """
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient fundamentals data"}

    details = []
    raw_score = 0  # We'll accumulate up to 6 points, then scale to 0–10

    # 1) Debt-to-Equity
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    eq_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]
    if debt_values and eq_values and len(debt_values) == len(eq_values) and len(debt_values) > 0:
        recent_debt = debt_values[0]
        recent_equity = eq_values[0] if eq_values[0] else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.5:
            raw_score += 2
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 1.0:
            raw_score += 1
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")
    else:
        details.append("No consistent debt/equity data available.")

    # 2) Operating Margin
    om_values = [fi.operating_margin for fi in financial_line_items if fi.operating_margin is not None]
    if om_values:
        om_recent = om_values[0]
        if om_recent > 0.20:
            raw_score += 2
            details.append(f"Strong operating margin: {om_recent:.1%}")
        elif om_recent > 0.10:
            raw_score += 1
            details.append(f"Moderate operating margin: {om_recent:.1%}")
        else:
            details.append(f"Low operating margin: {om_recent:.1%}")
    else:
        details.append("No operating margin data available.")

    # 3) Positive Free Cash Flow
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]
    if fcf_values and fcf_values[0] is not None:
        if fcf_values[0] > 0:
            raw_score += 2
            details.append(f"Positive free cash flow: {fcf_values[0]:,.0f}")
        else:
            details.append(f"Recent FCF is negative: {fcf_values[0]:,.0f}")
    else:
        details.append("No free cash flow data available.")

    # raw_score up to 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_lynch_valuation(financial_line_items: list, market_cap: float | None) -> dict:
    """
    Peter Lynch's approach to 'Growth at a Reasonable Price' (GARP):
      - Emphasize the PEG ratio: (P/E) / Growth Rate
      - Also consider a basic P/E if PEG is unavailable
    A PEG < 1 is very attractive; 1-2 is fair; >2 is expensive.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    details = []
    raw_score = 0

    # Gather data for P/E
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]

    # Approximate P/E via (market cap / net income) if net income is positive
    pe_ratio = None
    if net_incomes and net_incomes[0] and net_incomes[0] > 0:
        pe_ratio = market_cap / net_incomes[0]
        details.append(f"Estimated P/E: {pe_ratio:.2f}")
    else:
        details.append("No positive net income => can't compute approximate P/E")

    # If we have at least 2 EPS data points, let's estimate growth
    eps_growth_rate = None
    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        older_eps = eps_values[-1]
        if older_eps > 0:
            eps_growth_rate = (latest_eps - older_eps) / older_eps
            details.append(f"Approx EPS growth rate: {eps_growth_rate:.1%}")
        else:
            details.append("Cannot compute EPS growth rate (older EPS <= 0)")
    else:
        details.append("Not enough EPS data to compute growth rate")

    # Compute PEG if possible
    peg_ratio = None
    if pe_ratio and eps_growth_rate and eps_growth_rate > 0:
        # Peg ratio typically uses a percentage growth rate
        # So if growth rate is 0.25, we treat it as 25 for the formula => PE / 25
        # Alternatively, some treat it as 0.25 => we do (PE / (0.25 * 100)).
        # Implementation can vary, but let's do a standard approach: PEG = PE / (Growth * 100).
        peg_ratio = pe_ratio / (eps_growth_rate * 100)
        details.append(f"PEG ratio: {peg_ratio:.2f}")

    # Scoring logic:
    #   - P/E < 15 => +2, < 25 => +1
    #   - PEG < 1 => +3, < 2 => +2, < 3 => +1
    if pe_ratio is not None:
        if pe_ratio < 15:
            raw_score += 2
        elif pe_ratio < 25:
            raw_score += 1

    if peg_ratio is not None:
        if peg_ratio < 1:
            raw_score += 3
        elif peg_ratio < 2:
            raw_score += 2
        elif peg_ratio < 3:
            raw_score += 1

    final_score = min(10, (raw_score / 5) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_sentiment(news_items: list) -> dict:
    """
    Basic news sentiment check. Negative headlines weigh on the final score.
    """
    if not news_items:
        return {"score": 5, "details": "No news data; default to neutral sentiment"}

    negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
    negative_count = 0
    for news in news_items:
        title_lower = (news.title or "").lower()
        if any(word in title_lower for word in negative_keywords):
            negative_count += 1

    details = []
    if negative_count > len(news_items) * 0.3:
        # More than 30% negative => somewhat bearish => 3/10
        score = 3
        details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
    elif negative_count > 0:
        # Some negativity => 6/10
        score = 6
        details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
    else:
        # Mostly positive => 8/10
        score = 8
        details.append("Mostly positive or neutral headlines")

    return {"score": score, "details": "; ".join(details)}


def analyze_insider_activity(insider_trades: list) -> dict:
    """
    Simple insider-trade analysis:
      - If there's heavy insider buying, it's a positive sign.
      - If there's mostly selling, it's a negative sign.
      - Otherwise, neutral.
    """
    # Default 5 (neutral)
    score = 5
    details = []

    if not insider_trades:
        details.append("No insider trades data; defaulting to neutral")
        return {"score": score, "details": "; ".join(details)}

    buys, sells = 0, 0
    for trade in insider_trades:
        if trade.transaction_shares is not None:
            if trade.transaction_shares > 0:
                buys += 1
            elif trade.transaction_shares < 0:
                sells += 1

    total = buys + sells
    if total == 0:
        details.append("No significant buy/sell transactions found; neutral stance")
        return {"score": score, "details": "; ".join(details)}

    buy_ratio = buys / total
    if buy_ratio > 0.7:
        # Heavy buying => +3 => total 8
        score = 8
        details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
    elif buy_ratio > 0.4:
        # Some buying => +1 => total 6
        score = 6
        details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
    else:
        # Mostly selling => -1 => total 4
        score = 4
        details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

    return {"score": score, "details": "; ".join(details)}


def generate_lynch_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> PeterLynchSignal:
    """
    Generates a final JSON signal in Peter Lynch's voice & style.
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是彼得·林奇（Peter Lynch）风格的AI投资分析师，请严格依据林奇的投资理念给出投资信号：

1. 投资于你能理解的企业（“投资你熟悉的”），关注生活中常见、易懂的商业模式。
2. 强调合理价格下的成长（GARP），PEG比率是核心指标。
3. 寻找“十倍股”潜力公司（长期业绩和股价有望大幅增长）。
4. 偏好收入和盈利持续增长、负债适度、管理层稳健的公司。
5. 避免复杂、杠杆高、故事过于花哨的企业。
6. 风格务实、接地气，善于用生活化视角发现投资机会。

推理时请做到：
- 用通俗语言讲清公司的成长故事和核心逻辑。
- 分析收入、EPS成长、PEG比率、资产负债表等关键指标。
- 如有“十倍股”潜力请明确指出。
- 结合生活观察举例（如“如果我家人都在用这个产品……”）。
- 总结优缺点，结论明确（bullish、bearish、neutral）。

请严格按如下JSON格式返回：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "reasoning": "string"
}}
"""
            ),
            (
                "human",
                """请根据以下分析数据，生成一份彼得·林奇风格的投资信号：

{ticker}的分析数据：
{analysis_data}

请严格按如下JSON格式返回：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "reasoning": "string"
}}
"""
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return PeterLynchSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis; defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PeterLynchSignal,
        agent_name="peter_lynch_agent",
        default_factory=create_default_signal,
    )
