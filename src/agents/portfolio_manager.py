import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_manager", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_manager", None, "Generating trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_manager",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status("portfolio_manager", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""

    # 兜底：如果所有ticker的signals_by_ticker[ticker]为空或全为neutral，直接返回hold
    all_empty = True
    for ticker in tickers:
        sigs = signals_by_ticker.get(ticker, {})
        if sigs:
            # 只要有一个agent信号不是neutral就不算全空
            if any(sig.get("signal") not in ["neutral", None] for sig in sigs.values()):
                all_empty = False
                break
    if all_empty:
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=0.0,
                    reasoning="无有效信号，自动保持观望"
                ) for ticker in tickers
            }
        )

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一名投资组合经理，需要基于多个股票的分析信号做出最终交易决策。

交易规则：
- 多头操作：
  * 只有在有可用现金时才能买入
  * 只有当前持有该股票多头仓位时才能卖出
  * 卖出数量不得超过当前多头持仓
  * 买入数量不得超过该股票的最大可买数量（max_shares）

- 空头操作：
  * 只有在有可用保证金时才能做空（持仓价值 × 保证金比例）
  * 只有当前持有该股票空头仓位时才能回补
  * 回补数量不得超过当前空头持仓
  * 做空数量必须满足保证金要求

- max_shares 已预先计算好，确保不超过持仓限制
- 根据信号同时考虑多头和空头机会
- 始终保持合理的风险管理，兼顾多头和空头敞口

可用操作：
- "buy"：开仓或加仓多头
- "sell"：平仓或减仓多头
- "short"：开仓或加仓空头
- "cover"：平仓或减仓空头
- "hold"：不操作

输入说明：
- signals_by_ticker：股票代码到信号的字典
- max_shares：每只股票允许的最大持仓数量
- portfolio_cash：当前投资组合现金
- portfolio_positions：当前持仓（包括多头和空头）
- current_prices：每只股票的当前价格
- margin_requirement：当前空头保证金比例（如0.5表示50%）
- total_margin_used：当前已用保证金总额
""",
            ),
            (
                "human",
                """请根据团队的分析结果，为每只股票做出交易决策。

各股票的信号如下：
{signals_by_ticker}

当前价格：
{current_prices}

最大可买入数量：
{max_shares}

投资组合现金：{portfolio_cash}
当前持仓：{portfolio_positions}
当前保证金比例：{margin_requirement}
已用保证金总额：{total_margin_used}

请严格按如下JSON结构输出：
{{
  "decisions": {{
    "TICKER1": {{
      "action": "buy/sell/short/cover/hold",
      "quantity": 整数,
      "confidence": 0-100之间的浮点数,
      "reasoning": "string"
    }},
    "TICKER2": {{
      ...
    }},
    ...
  }}
}}
""",
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
            "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error in portfolio management, defaulting to hold") for ticker in tickers})

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=PortfolioManagerOutput, agent_name="portfolio_manager", default_factory=create_default_portfolio_output)
