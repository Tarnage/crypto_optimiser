import numpy as np
from core.bot import crossover_signals

FEE = 0.03
START_USD = 1000.0

def evaluate(theta, prices_full):
    """
    Enhanced evaluate function that returns final USD and detailed metrics:
    - trades: total completed trades
    - num_buys / num_sells
    - return: final USD / START_USD - 1
    - max_drawdown: maximum peak-to-trough drawdown
    - win_rate: fraction of trades with positive return
    - avg_trade_return: mean return per trade
    - trade_info: list of dicts per trade with entry/exit indices, prices, profit, pct_profit, duration
    - time_in_market: fraction of periods holding BTC
    - equity_curve: list of equity over time
    """
    buy_idx, sell_idx, tail_len = crossover_signals(prices_full, theta)
    price = prices_full[-tail_len:]
    n = tail_len

    holdings_usd = START_USD
    holdings_btc = 0.0
    trade_count = 0
    num_buys = 0
    num_sells = 0

    equity_curve = []
    trade_info = []
    in_market = np.zeros(n, dtype=bool)

    entry_idx = None
    entry_usd = None

    # simulate and log
    for t in range(n):
        # record equity
        equity = holdings_usd + holdings_btc * price[t]
        equity_curve.append(equity)

        # buy signal
        if buy_idx.size and t == buy_idx[0]:
            if holdings_usd > 0:
                cost_usd = holdings_usd
                btc_amount = cost_usd * (1 - FEE) / price[t]
                holdings_btc = btc_amount
                holdings_usd = 0.0

                entry_idx = t
                entry_usd = cost_usd
                trade_count += 1
                num_buys += 1
            buy_idx = buy_idx[1:]

        # mark in market
        if holdings_btc > 0:
            in_market[t] = True

        # sell signal
        if sell_idx.size and t == sell_idx[0]:
            if holdings_btc > 0 and entry_idx is not None:
                exit_usd = holdings_btc * price[t] * (1 - FEE)
                profit = exit_usd - entry_usd
                pct_profit = profit / entry_usd
                duration = t - entry_idx

                trade_info.append({
                    "entry_idx": entry_idx,
                    "exit_idx": t,
                    "entry_price": price[entry_idx],
                    "exit_price": price[t],
                    "profit": profit,
                    "pct_profit": pct_profit,
                    "duration": duration
                })

                holdings_usd = exit_usd
                holdings_btc = 0.0
                trade_count += 1
                num_sells += 1
                entry_idx = None
                entry_usd = None
            sell_idx = sell_idx[1:]

    # final close-out
    if holdings_btc > 0 and entry_idx is not None:
        t = n - 1
        exit_usd = holdings_btc * price[t] * (1 - FEE)
        profit = exit_usd - entry_usd
        pct_profit = profit / entry_usd
        duration = t - entry_idx
        trade_info.append({
            "entry_idx": entry_idx,
            "exit_idx": t,
            "entry_price": price[entry_idx],
            "exit_price": price[t],
            "profit": profit,
            "pct_profit": pct_profit,
            "duration": duration
        })
        holdings_usd = exit_usd
        holdings_btc = 0.0
        trade_count += 1
        num_sells += 1
        in_market[t] = True
        equity_curve.append(holdings_usd)

    # metrics
    returns = np.array([tr["pct_profit"] for tr in trade_info])
    win_rate = np.mean(returns > 0) if returns.size else np.nan
    avg_trade_return = np.mean(returns) if returns.size else np.nan

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    max_drawdown = np.max(drawdowns) if drawdowns.size else np.nan

    time_in_market = in_market.mean()

    total_return = holdings_usd / START_USD - 1.0

    aux = {
        "trades": trade_count,
        "num_buys": num_buys,
        "num_sells": num_sells,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "time_in_market": time_in_market,
        "trade_info": trade_info,
        # "equity_curve": equity_curve
    }
    return holdings_usd, aux
