# 📈 Trading Bot README

This project implements a **backtestable, evaluation-ready crossover trading bot** for cryptocurrency markets (tested on BTC daily data).  
The bot logic, fitness evaluation, and plotting are fully modular and optimised for testing with nature-inspired optimisation algorithms (CAB, PSO, GWO, etc.).

---

## ⚙️ Core Functionality

### Bot logic

The bot uses a **two-signal crossover** strategy:

- **Fast signal**: moving average with shorter window (e.g., 10 days).
- **Slow signal**: moving average with longer window (e.g., 60 days).

When the **fast signal crosses above** the slow signal → **Buy**.  
When the **fast signal crosses below** the slow signal → **Sell**.

All trades:
- Use 100% of available USD to buy BTC, or 100% of BTC to sell into USD.
- Apply a **3% transaction fee** per trade (both entry and exit).
- **Force close** any open BTC position at the last available price.

---

## 🧠 Moving Average Types

| Code | Type | Description |
|------|------|-------------|
| 0 | SMA | Simple Moving Average (equal weights) |
| 1 | LMA | Linear Moving Average (descending weights) |
| 2 | EMA | Exponential Moving Average (decay with α) |
| 3 | MACD | Moving Average Convergence Divergence |

These are generated using `make_signal(prices, window, type, alpha)`.

---

## 🏗️ Key Components

| File | Purpose |
|------|---------|
| `core/bot.py` | Crossover signal generation, MA filters, and helpers. |
| `evaluate(theta, prices_full)` | Main backtest loop. Executes trades, tracks equity, computes detailed metrics. |
| `foward_test.ipynb` | Plots Close price, fast/slow indicators, and buy/sell points for each bot configuration. |

---

## 📊 Metrics Collected

Each run of `evaluate` returns:

- **Final USD balance** (`holdings_usd`)
- **Trade statistics** (in a dictionary `aux`):
  - `trades`: total number of buy/sell events
  - `num_buys`, `num_sells`
  - `total_return`: (% gain/loss over starting USD)
  - `max_drawdown`: max peak-to-trough decline
  - `win_rate`: fraction of profitable trades
  - `avg_trade_return`: mean % return per trade
  - `time_in_market`: % of time holding BTC
  - `trade_info`: detailed list (entry, exit, profit, duration)

This structure makes it easy to compare different algorithmic optimisers fairly.

---

## 📦 Inputs / Outputs Summary

| Component | Input | Output |
|-----------|-------|--------|
| `evaluate(theta, prices)` | Bot config and close prices | Final USD, metrics dictionary |
| `crossover_signals(prices, theta)` | Close prices and config | Arrays of buy and sell indices |
| `plot_crossovers(best_df, prices_test, dates_test)` | Best configs + test prices | Crossover plots |

---

## 🧹 Assumptions & Notes

- Start capital: **$1000 USD**.
- Trade fees: **3%** per transaction.
- "All-in" trading: **no position sizing or leverage**.
- Crossover signals must be aligned correctly via moving average padding (mirror technique).
- No slippage model: trades execute at close price of the current candle.

---

## 🚀 Developer Quick Start Checklist

1. **Prepare your data**  
   - CSV with columns: `Date`, `Close`.
   - Use daily candles (can extend to hourly later if needed).

2. **Train the bot**  
   - Use any optimiser (CAB, PSO, Random Search) to propose `theta = [d₁, t₁, α₁, d₂, t₂, α₂]`.
   - Pass `theta` to `evaluate(theta, prices_train)`.

3. **Log the results**  
   - Save final USD balance and `aux` metrics per run.
   - Track `theta`, seed, and algorithm name.

4. **Test the best bots**  
   - Evaluate top configurations on the `prices_test` slice.
   - Record out-of-sample performance.

5. **Visualise the results**  
   - Call `plot_crossovers(best_df, prices_test, dates_test)`.
   - Inspect buy/sell points, fast/slow indicator lines, and general signal quality.

---

## 🧠 Quick Definitions

| Term | Meaning |
|------|--------|
| **Equity curve** | Time series of total USD value at each timestep. |
| **Max drawdown** | Biggest relative loss from a peak in equity. |
| **Win rate** | % of trades that closed profitably. |
| **Time in market** | % of periods where a BTC position was held. |

---

## ✅ Final Thoughts


---

