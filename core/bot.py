from core.filters import sma, lma, ema, macd
import numpy as np

TYPE_MAP = {0: "sma", 1: "lma", 2: "ema", 3: "macd"}

def make_signal(prices, window, typ, alpha):
    if typ == 0:        # SMA
        return sma(prices, window)
    if typ == 1:        # LMA
        return lma(prices, window)
    if typ == 2:        # EMA
        return ema(prices, window, alpha)
    if typ == 3:        # MACD
        return macd(prices, window, alpha) # TODO: not sure if this is correct
    raise ValueError("Unknown type")

def crossover_signals(prices, theta):
    d1, t1, a1, d2, t2, a2, buy_delay, sell_delay = theta

    d1, d2     = int(np.round(d1)), int(np.round(np.maximum(d2, d1 + 1)))
    t1, t2     = int(np.round(t1)), int(np.round(t2))
    buy_delay      = int(np.round(buy_delay))
    sell_delay      = int(np.round(sell_delay))

    sig_fast   = make_signal(prices, d1, t1, a1)
    sig_slow   = make_signal(prices, d2, t2, a2)

    # Align lengths (take common tail)
    min_len    = min(len(sig_fast), len(sig_slow))
    fast_tail  = sig_fast[-min_len:]
    slow_tail  = sig_slow[-min_len:]
    price_tail = prices[-min_len:]

    diff       = fast_tail - slow_tail
    cross      = np.sign(diff)
    cross_diff = np.diff(cross, prepend=cross[0])

    raw_buy_idx = np.where(cross_diff > 0)[0]
    raw_sell_idx = np.where(cross_diff < 0)[0]

    buy_idx = []
    sell_idx = []

    for i in raw_buy_idx:
        j = i + buy_delay
        if j < len(price_tail):
            if price_tail[j] > price_tail[i]:
                buy_idx.append(i)
    for i in raw_sell_idx:
        j = i + sell_delay
        if j < len(price_tail):
            if price_tail[j] < price_tail[i]:
                sell_idx.append(i)

    # Final clipping to prevent out-of-range indices
    buy_idx = np.array(buy_idx)
    sell_idx = np.array(sell_idx)
    buy_idx = buy_idx[(buy_idx >= 0) & (buy_idx < len(price_tail))]
    sell_idx = sell_idx[(sell_idx >= 0) & (sell_idx < len(price_tail))]

    return buy_idx, sell_idx, min_len
