import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("Downloading Stock Data...")
# We use Apple stock as example - you can change to TCS.NS, INFY.NS etc.
stock = yf.download("AAPL", period="1y", auto_adjust=True)
stock = stock[['Close']].dropna()

# Create feature: predict next day price using today's price
stock['Prediction'] = stock['Close'].shift(-1)
X = stock[['Close']][:-1]
y = stock['Prediction'][:-1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print(f"Model Accuracy - MSE: {mse}")

# Predict next 7 days (using last price)
last_price = stock[['Close']].iloc[-1].values[0]
future_prices = []
for i in range(7):
    next_price = model.predict([[last_price]])[0]
    future_prices.append(next_price)
    last_price = next_price

print("Next 7 days predicted prices:", future_prices)

# Graph
plt.figure(figsize=(10,5))
plt.plot(stock['Close'][-60:], label='Actual Price (Last 60 days)')
plt.plot(pd.date_range(stock.index[-1], periods=7, freq='D'), future_prices, label='Predicted (Next 7 days)', marker='o')
plt.title('Stock Price Predictor - AAPL')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.legend()
plt.savefig('output.png')
print("Graph saved as output.png")
