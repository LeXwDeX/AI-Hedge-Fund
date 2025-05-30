from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from .analysts import ANALYST_ORDER
import os
import json

console = Console()


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    用 rich.table 美观输出多 ticker 的分析和决策，自动适配中英文宽度。
    """
    decisions = result.get("decisions")
    if not decisions:
        console.print("[red]No trading decisions available[/red]")
        return

    for ticker, decision in decisions.items():
        console.print(f"\n[bold white]对 [cyan]{ticker}[/cyan] 的分析[/bold white]")
        console.print(f"[bold white]{'=' * 50}[/bold white]")

        # AGENT 分析表
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("代理人", style="cyan", no_wrap=True)
        table.add_column("信号", style="bold", justify="center")
        table.add_column("置信度", style="white", justify="right")
        table.add_column("理由", style="white", overflow="fold")

        agent_signals = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
            if agent == "risk_management_agent":
                continue
            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)
            signal_color = {
                "BULLISH": "green",
                "BEARISH": "red",
                "NEUTRAL": "yellow",
            }.get(signal_type, "white")
            # Reasoning
            reasoning = signal.get("reasoning", "")
            if isinstance(reasoning, dict):
                reasoning = json.dumps(reasoning, ensure_ascii=False, indent=2)
            elif not isinstance(reasoning, str):
                reasoning = str(reasoning)
            agent_signals.append(
                (agent_name, f"[{signal_color}]{signal_type}[/{signal_color}]", f"{confidence:.1f}%", reasoning)
            )
        agent_signals = sort_agent_signals(agent_signals)
        for row in agent_signals:
            table.add_row(*row)
        console.print(f"\n[bold white]代理人分析:[/bold white] [cyan]{ticker}[/cyan]")
        console.print(table)

        # 决策表
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": "green",
            "SELL": "red",
            "HOLD": "yellow",
            "COVER": "green",
            "SHORT": "red",
        }.get(action, "white")
        decision_table = Table(show_header=False, box=None)
        decision_table.add_row("操作", f"[{action_color}]{action}[/{action_color}]")
        decision_table.add_row("数量", f"[{action_color}]{decision.get('quantity')}[/{action_color}]")
        decision_table.add_row("置信度", f"[white]{decision.get('confidence'):.1f}%[/white]")
        reasoning = decision.get("reasoning", "")
        if isinstance(reasoning, dict):
            reasoning = json.dumps(reasoning, ensure_ascii=False, indent=2)
        elif not isinstance(reasoning, str):
            reasoning = str(reasoning)
        decision_table.add_row("理由", f"[white]{reasoning}[/white]")
        console.print(f"\n[bold white]交易决策:[/bold white] [cyan]{ticker}[/cyan]")
        console.print(decision_table)

    # Portfolio Summary
        console.print(f"\n[bold white]投资组合摘要:[/bold white]")
    portfolio_table = Table(show_header=True, header_style="bold magenta", box=None)
    portfolio_table.add_column("代码", style="cyan", no_wrap=True)
    portfolio_table.add_column("操作", style="bold", justify="center")
    portfolio_table.add_column("数量", style="white", justify="right")
    portfolio_table.add_column("置信度", style="white", justify="right")
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": "green",
            "SELL": "red",
            "HOLD": "yellow",
            "COVER": "green",
            "SHORT": "red",
        }.get(action, "white")
        portfolio_table.add_row(
            f"[cyan]{ticker}[/cyan]",
            f"[{action_color}]{action}[/{action_color}]",
            f"[{action_color}]{decision.get('quantity')}[/{action_color}]",
            f"[white]{decision.get('confidence'):.1f}%[/white]",
        )
    console.print(portfolio_table)

    # Portfolio Manager's reasoning
    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get("reasoning"):
            portfolio_manager_reasoning = decision.get("reasoning")
            break
    if portfolio_manager_reasoning:
        if isinstance(portfolio_manager_reasoning, dict):
            reasoning_str = json.dumps(portfolio_manager_reasoning, ensure_ascii=False, indent=2)
        else:
            reasoning_str = str(portfolio_manager_reasoning)
        console.print(f"\n[bold white]投资组合策略:[/bold white]")
        console.print(f"[cyan]{reasoning_str}[/cyan]")


def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    
    # Display latest portfolio summary
    if summary_rows:
        latest_summary = summary_rows[-1]
        print(f"\n{Fore.WHITE}{Style.BRIGHT}投资组合摘要:{Style.RESET_ALL}")

        # Extract values and remove commas before converting to float
        cash_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest_summary[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Return: {latest_summary[9]}")
        
        # Display performance metrics if available
        if latest_summary[10]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest_summary[10]}")
        if latest_summary[11]:  # Sortino ratio
            print(f"Sortino Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Max drawdown
            print(f"Max Drawdown: {latest_summary[12]}")

    # Add vertical spacing
    print("\n" * 2)

    # Print the table with just ticker rows
    print(
            ticker_rows,
            headers=[
                "日期",
                "代码",
                "操作",
                "数量",
                "价格",
                "持股",
                "持仓价值",
                "看涨",
                "看跌",
                "中性",
            ],
            tablefmt="grid",
            colalign=(
                "left",  # Date
                "left",  # Ticker
                "center",  # Action
                "right",  # Quantity
                "right",  # Price
                "right",  # Shares
                "right",  # Position Value
                "right",  # Bullish
                "right",  # Bearish
                "right",  # Neutral
            ),
    )

    # Add vertical spacing
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    shares_owned: float,
    position_value: float,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    # Color the action
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}",
        ]
