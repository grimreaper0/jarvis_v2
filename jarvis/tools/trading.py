"""Trading tools for LangGraph agents."""
from langchain_core.tools import tool


@tool
def get_market_data(symbol: str, timeframe: str = "1Min") -> dict:
    """Fetch latest market data for a symbol from Alpaca.

    Args:
        symbol: Stock ticker (e.g. 'AAPL')
        timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')

    Returns:
        Dict with OHLCV bars and current price
    """
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": [],
        "current_price": 0.0,
        "note": "Implement with Alpaca market data API",
    }


@tool
def get_account_info() -> dict:
    """Get current Alpaca account info: balance, buying power, positions.

    Returns:
        Dict with account balance, positions, and buying power
    """
    return {
        "buying_power": 0.0,
        "portfolio_value": 0.0,
        "positions": [],
        "note": "Implement with Alpaca REST API",
    }


@tool
def calculate_vwap(symbol: str, bars: list[dict]) -> float:
    """Calculate VWAP (Volume Weighted Average Price) from bar data.

    Args:
        symbol: Stock ticker
        bars: List of OHLCV bars with volume

    Returns:
        VWAP value as float
    """
    if not bars:
        return 0.0
    total_pv = sum(((b["high"] + b["low"] + b["close"]) / 3) * b["volume"] for b in bars)
    total_vol = sum(b["volume"] for b in bars)
    return total_pv / total_vol if total_vol else 0.0
