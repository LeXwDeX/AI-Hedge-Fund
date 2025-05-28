from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def bill_ackman_agent(state: AgentState):
    """
    只用AKSHARE行情数据（ak.stock_zh_a_hist字段），做统一技术分析（动量/波动/均线/换手率等）。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    ackman_analysis = {}

    for ticker in tickers:
        progress.update_status("bill_ackman_agent", ticker, "获取AKSHARE行情数据")
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "score": 0, "max_score": 3, "details": "无行情数据"}
            continue
        # 只取最近30条
        prices = prices[-30:]

        # 直接将行情数据和技术分析提示词交给LLM
        progress.update_status("bill_ackman_agent", ticker, "生成技术分析信号")
        ackman_output = generate_ackman_technical_output(
            ticker=ticker,
            prices=prices,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # A股不能做空，bearish时给出替代建议
        reasoning = ackman_output.reasoning
        if ackman_output.signal == "bearish":
            reasoning = f"{reasoning}（A股市场不能做空，建议观望或空仓，避免盲目操作。）"

        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": reasoning
        }

        progress.update_status("bill_ackman_agent", ticker, "Done", analysis=ackman_output.reasoning)

    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name="bill_ackman_agent"
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    state["data"]["analyst_signals"]["bill_ackman_agent"] = ackman_analysis

    progress.update_status("bill_ackman_agent", None, "Done")

    return {
        "messages": [message],
        "data": state["data"]
    }

def generate_ackman_technical_output(
    ticker: str,
    prices: list[dict],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
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

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BillAckmanSignal,
        agent_name="bill_ackman_agent",
        default_factory=create_default_bill_ackman_signal,
    )
