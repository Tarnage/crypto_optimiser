# core/filters.py
import numpy as np

# ---------- helpers ----------------------------------------------------------
def _pad(signal: np.ndarray, N: int) -> np.ndarray:
    """
    Mirror-pad the first N-1 samples so that the window is full for the
    first real point (matches the specification's 'flip' trick).
    """
    if N <= 1:
        return signal
    padding = -np.flip(signal[1:N])     # mirror, then invert sign as per spec
    return np.concatenate((padding, signal))


def _wma(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Generic weighted moving average via 1-D convolution.
    Returns length == len(signal) (same as original candles).
    """
    N = len(kernel)
    padded = _pad(signal, N)
    # 'valid' => only positions where the kernel fully overlaps the padded array
    return np.convolve(padded, kernel, mode="valid")


# ---------- filters ----------------------------------------------------------
def sma(signal: np.ndarray, N: int) -> np.ndarray:
    """
    Simple Moving Average (Eq 1).
    kernel = [1/N, 1/N, …]
    """
    kernel = np.ones(N, dtype=float) / N
    return _wma(signal, kernel)


def lma(signal: np.ndarray, N: int) -> np.ndarray:
    """
    Linear-Weighted Moving Average (Eq 4).
    Triangular weights descending to zero, normalised by 2/(N+1).
    """
    # descending integers N, …, 1
    w = np.arange(N, 0, -1, dtype=float)
    kernel = w / w.sum()               # normalise exactly to Σ=1
    return _wma(signal, kernel)


def ema(signal: np.ndarray, N: int, alpha: float) -> np.ndarray:
    """
    Exponential Moving Average (Eq 5, truncated window).
    We build the kernel: k_i = α (1-α)^i, i = 0 … N-1,
    then normalise so Σ=1.  Larger α ⇒ sharper decay.
    """
    idx = np.arange(N, dtype=float)
    w   = alpha * (1 - alpha) ** idx
    kernel = w / w.sum()
    return _wma(signal, kernel)

# TODO: not sure if this is correct
def macd(signal: np.ndarray, N: int, alpha: float) -> np.ndarray:
    """
    Moving Average Convergence Divergence (Eq 6).
    MACD = EMA(α) - EMA(β), where α = short window, β = long window.
    """
    fast_N = int(N // 2)
    slow_N = int(N)

    # Compute EMAs
    ema_fast = ema(signal, fast_N, alpha)
    ema_slow = ema(signal, slow_N, alpha)

    # Align lengths by taking the common tail
    min_len = min(len(ema_fast), len(ema_slow))
    return ema_fast[-min_len:] - ema_slow[-min_len:]