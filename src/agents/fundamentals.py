from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import json

from src.tools.api import get_financial_metrics


##### Fundamental Agent #####
def fundamentals_analyst_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_analyst_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_analyst_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # 只保留盈利能力分析
        reasoning = {}

        progress.update_status("fundamentals_analyst_agent", ticker, "Analyzing profitability")
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        thresholds = [
            (return_on_equity, 0.15),
            (net_margin, 0.20),
            (operating_margin, 0.15),
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        if profitability_score >= 2:
            overall_signal = "bullish"
        elif profitability_score == 0:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        confidence = 100.0  # 只剩一个信号，置信度恒为100

        reasoning["profitability_signal"] = {
            "signal": overall_signal,
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " +
                       (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " +
                       (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_analyst_agent", ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_analyst_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_analyst_agent"] = fundamental_analysis

    progress.update_status("fundamentals_analyst_agent", None, "Done")
    
    return {
        "messages": [message],
        "data": data,
    }
