import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the dataset
filename = "TSLA.csv"  # Use your manually downloaded file
df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")

# Use only 'Close' prices for LSTM
data = df[['Close']].values  

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Use last 50 days to predict next
X, y = create_sequences(scaled_data, seq_length)

# Split data into training & testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Prepare for visualization
test_dates = df.index[split + seq_length + 1:]
actual_prices = df["Close"][split + seq_length + 1:]

# Create Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Tesla (TSLA) Stock Prediction with LSTM", style={'textAlign': 'center'}),

    dcc.Graph(id="stock-forecast")
])

@app.callback(
    Output("stock-forecast", "figure"),
    Input("stock-forecast", "id")
)
def update_stock_forecast(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), mode='lines', name='Predicted Prices', line=dict(dash='dot')))
    fig.update_layout(title="Tesla Stock Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
