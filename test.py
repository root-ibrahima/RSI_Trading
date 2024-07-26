import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf

# Generate sample data for demonstration purposes
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
prices = np.cumsum(np.random.randn(100)) + 100
open_prices = prices + np.random.randn(100)
high_prices = np.maximum(open_prices, prices + np.random.randn(100))
low_prices = np.minimum(open_prices, prices - np.random.randn(100))
close_prices = prices

# Create a DataFrame for mplfinance
data = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices
})
data.set_index('Date', inplace=True)

# Optimized Parabolic SAR parameters
def parabolic_sar(prices, af=0.02, max_af=0.2):
    n = len(prices)
    sar = np.zeros(n)
    trend = np.zeros(n)
    ep = prices[0]
    sar[0] = prices[0]
    af_factor = af
    long = True

    for i in range(1, n):
        if long:
            sar[i] = sar[i-1] + af_factor * (ep - sar[i-1])
            if prices[i] > ep:
                ep = prices[i]
                af_factor = min(af_factor + af, max_af)
            if prices[i] < sar[i]:
                long = False
                sar[i] = ep
                af_factor = af
                ep = prices[i]
        else:
            sar[i] = sar[i-1] - af_factor * (sar[i-1] - ep)
            if prices[i] < ep:
                ep = prices[i]
                af_factor = min(af_factor + af, max_af)
            if prices[i] > sar[i]:
                long = True
                sar[i] = ep
                af_factor = af
                ep = prices[i]
        trend[i] = long

    return sar, trend

sar, trend = parabolic_sar(close_prices)

# Optimized RSI parameters
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = calculate_rsi(close_prices)

# Ensure RSI and dates have the same length
rsi = rsi[~np.isnan(rsi)]  # Remove NaN values
dates_rsi = dates[1:len(rsi)+1]  # Adjust dates to match RSI length

# Identify buy and sell signals
buy_signals = (rsi < 30)
sell_signals = (rsi > 70)

# Ensure signals match the length of dates_rsi
buy_signals = buy_signals[:len(dates_rsi)]
sell_signals = sell_signals[:len(dates_rsi)]

# Adjust close_prices to match dates_rsi length
adjusted_close_prices = close_prices[1:len(dates_rsi)+1]

# Set the style to dark background
plt.style.use('dark_background')

# Plotting
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)

# Plot candlestick chart with Parabolic SAR
mpf.plot(data, type='candle', ax=ax1, style='charles', show_nontrading=True)
ax1.plot(dates, sar, color='red', marker='.', linestyle='None', label='Parabolic SAR', alpha=0.7)
for i in range(1, len(dates)):
    if trend[i] != trend[i-1]:
        ax1.axvline(x=dates[i], color='gray', linestyle='--', alpha=0.5)

# Add buy and sell signals to candlestick chart
ax1.scatter(dates_rsi[buy_signals], adjusted_close_prices[buy_signals], marker='^', color='lime', label='Buy Signal', alpha=1)
ax1.scatter(dates_rsi[sell_signals], adjusted_close_prices[sell_signals], marker='v', color='red', label='Sell Signal', alpha=1)

ax1.set_title('Parabolic SAR and Prices')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, color='gray')

# Plot RSI
ax2.plot(dates_rsi, rsi, label='RSI', color='lime')
ax2.axhline(70, color='red', linestyle='--', label='Overbought')
ax2.axhline(30, color='blue', linestyle='--', label='Oversold')
ax2.set_title('Relative Strength Index (RSI)')
ax2.set_xlabel('Date')
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(True, color='gray')

plt.tight_layout()
plt.show()
