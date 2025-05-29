from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.tools.api import get_market_cap, search_line_items
from src.data.baostock_service import get_financial_data
from src.utils.llm import call_llm
from src.utils.progress import progress

class FinancialMetrics(BaseModel):
    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    operating_margin: float | None = None
    current_ratio: float | None = None


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis for LLM reasoning
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching financial metrics")
        # Fetch required data
        # 使用Baostock获取财务数据
        metrics_df = get_financial_data(ticker, int(end_date[:4]), (int(end_date[5:7]) - 1) // 3 + 1)
        
        # 转换为FinancialMetrics对象
        if not metrics_df.empty:
            metrics = FinancialMetrics(
                return_on_equity=float(metrics_df['roeAvg'].iloc[0]),
                debt_to_equity=float(metrics_df['liabilityToAsset'].iloc[0]),
                operating_margin=float(metrics_df['npMargin'].iloc[0]),
                current_ratio=float(metrics_df['currentRatio'].iloc[0])
            )
        else:
            metrics = FinancialMetrics()

        progress.update_status("warren_buffett_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("warren_buffett_agent", ticker, "Getting market cap")
        # Get current market cap
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        # Analyze fundamentals
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # Calculate total score
        total_score = fundamental_analysis["score"] + consistency_analysis["score"] + moat_analysis["score"] + mgmt_analysis["score"]
        max_possible_score = 10 + moat_analysis["max_score"] + mgmt_analysis["max_score"]
        # fundamental_analysis + consistency combined were up to 10 points total
        # moat can add up to 3, mgmt can add up to 2, for example

        # Add margin of safety analysis if we have both intrinsic value and current price
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # Generate trading signal using a stricter margin-of-safety requirement
        # if fundamentals+moat+management are strong but margin_of_safety < 0.3, it's neutral
        # if fundamentals are very weak or margin_of_safety is severely negative -> bearish
        # else bullish
        if (total_score >= 0.7 * max_possible_score) and margin_of_safety and (margin_of_safety >= 0.3):
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            # negative margin of safety beyond -30% could be overpriced -> bearish
            signal = "bearish"
        else:
            signal = "neutral"

        # Combine all analysis results
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # Store analysis in consistent format with other agents
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,  # Normalize between 0 to 100
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done", analysis=buffett_output.reasoning)

    # Create the message
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    progress.update_status("warren_buffett_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not hasattr(metrics, 'return_on_equity'):
        return {"score": 0, "details": "Insufficient fundamental data"}

    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    if metrics.return_on_equity and metrics.return_on_equity > 0.15:  # 15% ROE threshold
        score += 2
        reasoning.append(f"Strong ROE of {metrics.return_on_equity:.1%}")
    elif metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    if metrics.debt_to_equity and metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    if metrics.operating_margin and metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    if metrics.current_ratio and metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # Check earnings growth trend
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate total growth rate from oldest to latest
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    # Normalize metrics to a list for analysis
    metrics_list = metrics if isinstance(metrics, list) else [metrics]

    # Require at least one period for analysis
    if len(metrics_list) < 1:
        return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    for m in metrics_list:
        if m.return_on_equity is not None:
            historical_roes.append(m.return_on_equity)
        if m.operating_margin is not None:
            historical_margins.append(m.operating_margin)

    # Check for stable or improving ROE
    if len(historical_roes) >= 3:
        stable_roe = all(r > 0.15 for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")

    # Check for stable or improving operating margin
    if len(historical_margins) >= 3:
        stable_margin = all(m > 0.15 for m in historical_margins)
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasoning.append("Operating margin not consistently above 15%")

    # If both are stable/improving, add an extra point
    if moat_score == 2:
        moat_score += 1
        reasoning.append("Both ROE and margin stability indicate a solid moat")

    return {
        "score": moat_score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        # Negative means the company spent money on buybacks
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        # Positive issuance means new shares => possible dilution
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # Check for any dividends
    if hasattr(latest, "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    if not financial_line_items or len(financial_line_items) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]

    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    # Estimate maintenance capex (typically 70-80% of total capex)
    maintenance_capex = capex * 0.75
    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"],
    }


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    if not financial_line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data for valuation"]}

    # Calculate owner earnings
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]

    # Get current market data
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding:
        return {"intrinsic_value": None, "details": ["Missing shares outstanding data"]}

    # Buffett's DCF assumptions (conservative approach)
    growth_rate = 0.05  # Conservative 5% growth
    discount_rate = 0.09  # Typical ~9% discount rate
    terminal_multiple = 12
    projection_years = 10

    # Sum of discounted future owner earnings
    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    # Terminal value
    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)

    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }


def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是沃伦·巴菲特风格的AI投资分析师，请严格依据巴菲特的投资原则给出投资信号：
- 能力圈：只投资你真正理解的企业
- 安全边际（>30%）：只有在价格大幅低于内在价值时才买入
- 经济护城河：寻找具有持久竞争优势的公司
- 优质管理层：偏好保守、以股东为中心的团队
- 财务稳健：低负债、高股本回报率
- 长期视角：投资企业而非仅仅投资股票
- 只有当基本面恶化或估值远超内在价值时才卖出

推理时请做到：
1. 详细说明影响你决策的关键因素（正反两面都要提及）
2. 强调公司如何契合或违背巴菲特的具体原则
3. 相关处请给出定量证据（如具体利润率、ROE、负债率等）
4. 以巴菲特风格总结投资机会
5. 用巴菲特的语气和对话风格表达

例如，看多时：“我尤其欣赏[具体优点]，这让我想起我们早期投资See's Candies时看到的[类似特质]……”
例如，看空时：“资本回报率的下滑让我联想到伯克希尔曾经的纺织业务，最终我们也因类似原因选择退出……”

请严格遵循以上指引。
""",
            ),
            (
                "human",
                """请根据以下数据，像巴菲特一样生成投资信号：

{ticker}的分析数据：
{analysis_data}

请严格按如下JSON格式返回交易信号：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100之间的浮点数,
  "reasoning": "string"
}}
""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        default_factory=create_default_warren_buffett_signal,
    )
