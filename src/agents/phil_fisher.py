from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def phil_fisher_agent(state: AgentState):
    """
    只用AKSHARE行情数据（ak.stock_zh_a_hist字段），做统一技术分析（动量/波动/均线/换手率等）。
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    fisher_analysis = {}

    for ticker in tickers:
        progress.update_status("phil_fisher_agent", ticker, "获取AKSHARE行情数据")
        prices = get_prices(ticker, "1990-01-01", end_date)
        if not prices:
            analysis_data[ticker] = {"signal": "neutral", "details": "无行情数据"}
            continue
        prices = prices[-30:]
        progress.update_status("phil_fisher_agent", ticker, "生成技术分析信号")
        fisher_output = generate_fisher_technical_output(
            ticker=ticker,
            prices=prices,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        # A股不能做空，bearish时给出替代建议
        reasoning = fisher_output.reasoning
        if fisher_output.signal == "bearish":
            reasoning = f"{reasoning}（A股市场不能做空，建议观望或空仓，避免盲目操作。）"

        fisher_analysis[ticker] = {
            "signal": fisher_output.signal,
            "confidence": fisher_output.confidence,
            "reasoning": reasoning
        }
        progress.update_status("phil_fisher_agent", ticker, "Done", analysis=fisher_output.reasoning)
    message = HumanMessage(content=json.dumps(fisher_analysis), name="phil_fisher_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(fisher_analysis, "Phil Fisher Agent")
    state["data"]["analyst_signals"]["phil_fisher_agent"] = fisher_analysis
    progress.update_status("phil_fisher_agent", None, "Done")
    return {"messages": [message], "data": state["data"]}

def generate_fisher_technical_output(
    ticker: str,
    prices: list[dict],
    model_name: str,
    model_provider: str,
) -> PhilFisherSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是技术分析专家。请只基于下方A股行情数据（包含日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率等），用你的技术分析知识（如趋势、动量、波动、均线、换手率、K线形态等）分析该股票当前的盘面特征，给出bullish（看多）、bearish（看空）、neutral（中性）信号，并详细说明理由。禁止参考任何财务、估值、成长等信息。\n数据如下（最近30日）：\n{prices}\n请严格按如下JSON格式返回：\n{{\n  \"signal\": \"bullish\" | \"bearish\" | \"neutral\",\n  \"confidence\": 0-100,\n  \"reasoning\": \"string\"\n}}"
        ),
    ])
    prompt = template.invoke({
        "prices": json.dumps(prices, ensure_ascii=False, indent=2),
        "ticker": ticker
    })
    def create_default_signal():
        return PhilFisherSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PhilFisherSignal,
        agent_name="phil_fisher_agent",
        default_factory=create_default_signal,
    )
