from .state import AnalystFinding

PROMPTS = {
    "fundamental": "You are a fundamental analyst. Assess valuation, profitability, growth, leverage.",
    "technical": "You are a technical analyst. Assess trend, momentum (RSI/MACD), Bollinger, support/resistance.",
    "risk": "You are a risk analyst. Assess beta, volatility, drawdown, Sharpe, VaR.",
    "valuation": "You are a valuation analyst. Assess DCF value, multiples, and a price target.",
}


def make_analyst_node(role: str, structured_factory):
    chain = structured_factory(AnalystFinding)

    def node(state):
        msgs = [{"role": "system", "content": PROMPTS[role]},
                {"role": "user", "content": state["metrics_block"] +
                 f"\n\nAnalyze {state['symbol']}. Cite each metric key you use."}]
        return {role: chain.invoke(msgs)}

    return node
