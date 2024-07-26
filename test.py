import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Générer des données d'exemple pour des fins de démonstration
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
prices = np.cumsum(np.random.randn(100)) + 100
open_prices = prices + np.random.randn(100)
high_prices = np.maximum(open_prices, prices + np.random.randn(100))
low_prices = np.minimum(open_prices, prices - np.random.randn(100))
close_prices = prices

# Créer un DataFrame pour mplfinance
data = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices
})
data.set_index('Date', inplace=True)

# Fonction pour calculer le Parabolic SAR
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

# Calcul du Parabolic SAR
sar, trend = parabolic_sar(close_prices)

# Fonction pour calculer le RSI
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
    rsi[:period] = np.nan  # RSI is undefined for the first `period` values
    return rsi

# Calcul du RSI
rsi = calculate_rsi(close_prices)

# Aligner les longueurs des séries temporelles
aligned_data = data.iloc[14:]  # Ignorer les 14 premiers jours où RSI n'est pas défini
aligned_sar = sar[14:]
aligned_rsi = rsi[14:]

# Extraire les caractéristiques et les labels
features = pd.DataFrame({
    'Close': aligned_data['Close'],
    'ParabolicSAR': aligned_sar,
    'RSI': aligned_rsi
})

# Créer la colonne de tendance basée sur le RSI
features['Trend'] = np.where(features['RSI'] < 30, 1, np.where(features['RSI'] > 70, -1, 0))

# Supprimer les valeurs NaN
features.dropna(inplace=True)

# Séparer les caractéristiques et les labels
X = features[['Close', 'ParabolicSAR', 'RSI']]
y = features['Trend']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir le modèle RandomForest
model = RandomForestClassifier(random_state=42)

# Définir les paramètres pour la recherche sur grille
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Recherche sur grille avec validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
best_model = grid_search.best_estimator_

# Prédictions
y_pred = best_model.predict(X_test)

# Évaluation
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualisation des signaux d'achat et de vente
buy_signals = X_test[y_pred == 1].index
sell_signals = X_test[y_pred == -1].index

# Convertir close_prices en pd.Series avec les dates comme index
close_prices_series = pd.Series(close_prices, index=dates)

# Ajuster les dates pour correspondre aux signaux
adjusted_close_prices = close_prices_series.loc[features.index]

# Plotting
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)

# Plot candlestick chart with Parabolic SAR
mpf.plot(aligned_data, type='candle', ax=ax1, style='charles', show_nontrading=True)
ax1.plot(aligned_data.index, aligned_sar, color='red', marker='.', linestyle='None', label='Parabolic SAR', alpha=0.7)
for i in range(1, len(aligned_data.index)):
    if trend[i+14] != trend[i+13]:  # Offset by 14 due to initial RSI NaNs
        ax1.axvline(x=aligned_data.index[i], color='gray', linestyle='--', alpha=0.5)

# Add buy and sell signals to candlestick chart
ax1.scatter(buy_signals, adjusted_close_prices[buy_signals], marker='^', color='lime', label='Buy Signal', alpha=1)
ax1.scatter(sell_signals, adjusted_close_prices[sell_signals], marker='v', color='red', label='Sell Signal', alpha=1)

ax1.set_title('Parabolic SAR and Prices with Buy/Sell Signals')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, color='gray')

# Plot RSI
ax2.plot(aligned_data.index, aligned_rsi, label='RSI', color='lime')
ax2.axhline(70, color='red', linestyle='--', label='Overbought')
ax2.axhline(30, color='blue', linestyle='--', label='Oversold')
ax2.set_title('Relative Strength Index (RSI)')
ax2.set_xlabel('Date')
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(True, color='gray')

plt.tight_layout()
plt.show()
