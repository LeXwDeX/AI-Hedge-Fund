from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

class AswathDamodaranSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def aswath_damodaran_agent(state: AgentState):
    """
    只用AKSHARE行情数据（ak.stock_zh_a_hist字段），做统一技术分析（动量/波动/均线/换手率等）。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    damodaran_signals = {}

    for ticker in tickers:
        progress.update_status("aswath_damodaran_agent", ticker, "获取AKSHARE行情数据")
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "score": 0, "max_score": 3, "details": "无行情数据"}
            continue
        # 只取最近30条
        prices = prices[-30:]

        # 直接将行情数据和技术分析提示词交给LLM
        progress.update_status("aswath_damodaran_agent", ticker, "生成技术分析信号")
        damodaran_output = generate_damodaran_technical_output(
            ticker=ticker,
            prices=prices,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # A股不能做空，bearish时给出替代建议
        output_dict = damodaran_output.model_dump()
        if output_dict.get("signal") == "bearish":
            output_dict["reasoning"] = f"{output_dict['reasoning']}（A股市场不能做空，建议观望或空仓，避免盲目操作。）"

        damodaran_signals[ticker] = output_dict

        progress.update_status("aswath_damodaran_agent", ticker, "Done", analysis=damodaran_output.reasoning)

    message = HumanMessage(content=json.dumps(damodaran_signals), name="aswath_damodaran_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(damodaran_signals, "Aswath Damodaran Agent")

    state["data"]["analyst_signals"]["aswath_damodaran_agent"] = damodaran_signals
    progress.update_status("aswath_damodaran_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}

def generate_damodaran_technical_output(
    ticker: str,
    prices: list[dict],
    model_name: str,
    model_provider: str,
) -> AswathDamodaranSignal:
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

    def default_signal():
        return AswathDamodaranSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error; defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=AswathDamodaranSignal,
        agent_name="aswath_damodaran_agent",
        default_factory=default_signal,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Helper analyses
# ────────────────────────────────────────────────────────────────────────────────
def analyze_growth_and_reinvestment(metrics: list, line_items: list) -> dict[str, any]:
    """
    Growth score (0‑4):
      +2  5‑yr CAGR of revenue > 8 %
      +1  5‑yr CAGR of revenue > 3 %
      +1  Positive FCFF growth over 5 yr
    Reinvestment efficiency (ROIC > WACC) adds +1
    """
    max_score = 4
    if len(metrics) < 2:
        return {"score": 0, "max_score": max_score, "details": "Insufficient history"}

    # Revenue CAGR (oldest to latest)
    revs = [m.revenue for m in reversed(metrics) if hasattr(m, "revenue") and m.revenue]
    if len(revs) >= 2 and revs[0] > 0:
        cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
    else:
        cagr = None

    score, details = 0, []

    if cagr is not None:
        if cagr > 0.08:
            score += 2
            details.append(f"Revenue CAGR {cagr:.1%} (> 8 %)")
        elif cagr > 0.03:
            score += 1
            details.append(f"Revenue CAGR {cagr:.1%} (> 3 %)")
        else:
            details.append(f"Sluggish revenue CAGR {cagr:.1%}")
    else:
        details.append("Revenue data incomplete")

    # FCFF growth (proxy: free_cash_flow trend)
    fcfs = [li.free_cash_flow for li in reversed(line_items) if li.free_cash_flow]
    if len(fcfs) >= 2 and fcfs[-1] > fcfs[0]:
        score += 1
        details.append("Positive FCFF growth")
    else:
        details.append("Flat or declining FCFF")

    # Reinvestment efficiency (ROIC vs. 10 % hurdle)
    latest = metrics[0]
    if latest.return_on_invested_capital and latest.return_on_invested_capital > 0.10:
        score += 1
        details.append(f"ROIC {latest.return_on_invested_capital:.1%} (> 10 %)")

    return {"score": score, "max_score": max_score, "details": "; ".join(details), "metrics": latest.model_dump()}


def analyze_risk_profile(metrics: list, line_items: list) -> dict[str, any]:
    """
    Risk score (0‑3):
      +1  Beta < 1.3
      +1  Debt/Equity < 1
      +1  Interest Coverage > 3×
    """
    max_score = 3
    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics"}

    latest = metrics[0]
    score, details = 0, []

    # Beta
    beta = getattr(latest, "beta", None)
    if beta is not None:
        if beta < 1.3:
            score += 1
            details.append(f"Beta {beta:.2f}")
        else:
            details.append(f"High beta {beta:.2f}")
    else:
        details.append("Beta NA")

    # Debt / Equity
    dte = getattr(latest, "debt_to_equity", None)
    if dte is not None:
        if dte < 1:
            score += 1
            details.append(f"D/E {dte:.1f}")
        else:
            details.append(f"High D/E {dte:.1f}")
    else:
        details.append("D/E NA")

    # Interest coverage
    ebit = getattr(latest, "ebit", None)
    interest = getattr(latest, "interest_expense", None)
    if ebit and interest and interest != 0:
        coverage = ebit / abs(interest)
        if coverage > 3:
            score += 1
            details.append(f"Interest coverage × {coverage:.1f}")
        else:
            details.append(f"Weak coverage × {coverage:.1f}")
    else:
        details.append("Interest coverage NA")

    # Compute cost of equity for later use
    cost_of_equity = estimate_cost_of_equity(beta)

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "beta": beta,
        "cost_of_equity": cost_of_equity,
    }


def analyze_relative_valuation(metrics: list) -> dict[str, any]:
    """
    Simple PE check vs. historical median (proxy since sector comps unavailable):
      +1 if TTM P/E < 70 % of 5‑yr median
      +0 if between 70 %‑130 %
      ‑1 if >130 %
    """
    max_score = 1
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": max_score, "details": "Insufficient P/E history"}

    pes = [m.price_to_earnings_ratio for m in metrics if m.price_to_earnings_ratio]
    if len(pes) < 5:
        return {"score": 0, "max_score": max_score, "details": "P/E data sparse"}

    ttm_pe = pes[0]
    median_pe = sorted(pes)[len(pes) // 2]

    if ttm_pe < 0.7 * median_pe:
        score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
    elif ttm_pe > 1.3 * median_pe:
        score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
    else:
        score, desc = 0, f"P/E inline with history"

    return {"score": score, "max_score": max_score, "details": desc}


# ────────────────────────────────────────────────────────────────────────────────
# Intrinsic value via FCFF DCF (Damodaran style)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_intrinsic_value_dcf(metrics: list, line_items: list, risk_analysis: dict) -> dict[str, any]:
    """
    FCFF DCF with:
      • Base FCFF = latest free cash flow
      • Growth = 5‑yr revenue CAGR (capped 12 %)
      • Fade linearly to terminal growth 2.5 % by year 10
      • Discount @ cost of equity (no debt split given data limitations)
    """
    if not metrics or len(metrics) < 2 or not line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data"]}

    latest_m = metrics[0]
    fcff0 = getattr(latest_m, "free_cash_flow", None)
    shares = getattr(line_items[0], "outstanding_shares", None)
    if not fcff0 or not shares:
        return {"intrinsic_value": None, "details": ["Missing FCFF or share count"]}

    # Growth assumptions
    revs = [m.revenue for m in reversed(metrics) if m.revenue]
    if len(revs) >= 2 and revs[0] > 0:
        base_growth = min((revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1, 0.12)
    else:
        base_growth = 0.04  # fallback

    terminal_growth = 0.025
    years = 10

    # Discount rate
    discount = risk_analysis.get("cost_of_equity") or 0.09

    # Project FCFF and discount
    pv_sum = 0.0
    g = base_growth
    g_step = (terminal_growth - base_growth) / (years - 1)
    for yr in range(1, years + 1):
        fcff_t = fcff0 * (1 + g)
        pv = fcff_t / (1 + discount) ** yr
        pv_sum += pv
        g += g_step

    # Terminal value (perpetuity with terminal growth)
    tv = (
        fcff0
        * (1 + terminal_growth)
        / (discount - terminal_growth)
        / (1 + discount) ** years
    )

    equity_value = pv_sum + tv
    intrinsic_per_share = equity_value / shares

    return {
        "intrinsic_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share,
        "assumptions": {
            "base_fcff": fcff0,
            "base_growth": base_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount,
            "projection_years": years,
        },
        "details": ["FCFF DCF completed"],
    }


def estimate_cost_of_equity(beta: float | None) -> float:
    """CAPM: r_e = r_f + β × ERP (use Damodaran’s long‑term averages)."""
    risk_free = 0.04          # 10‑yr US Treasury proxy
    erp = 0.05                # long‑run US equity risk premium
    beta = beta if beta is not None else 1.0
    return risk_free + beta * erp


# ────────────────────────────────────────────────────────────────────────────────
# LLM generation
# ────────────────────────────────────────────────────────────────────────────────
def generate_damodaran_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> AswathDamodaranSignal:
    """
    Ask the LLM to channel Prof. Damodaran’s analytical style:
      • Story → Numbers → Value narrative
      • Emphasize risk, growth, and cash‑flow assumptions
      • Cite cost of capital, implied MOS, and valuation cross‑checks
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是阿斯沃斯·达摩达兰（Aswath Damodaran），纽约大学斯特恩商学院金融学教授。
请用你的估值框架对美股公司给出交易信号。

表达风格需清晰、数据驱动：
  ◦ 先简要讲述公司的“故事”（定性分析）
  ◦ 将故事与关键数字驱动因素关联：收入增长、利润率、再投资、风险
  ◦ 最后给出价值结论：你的FCFF DCF估值、安全边际、相对估值校验
  ◦ 明确指出主要不确定性及其对价值的影响
只返回下方指定的JSON。""",
            ),
            (
                "human",
                """股票代码：{ticker}

分析数据：
{analysis_data}

请严格按如下JSON格式返回：
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": 0-100之间的浮点数,
  "reasoning": "string"
}}""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default_signal():
        return AswathDamodaranSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error; defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=AswathDamodaranSignal,
        agent_name="aswath_damodaran_agent",
        default_factory=default_signal,
    )
