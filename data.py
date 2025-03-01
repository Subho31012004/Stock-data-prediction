import yfinance as yf
import pandas as pd
import time

# Define stock symbol and filename
stock_symbol = "TSLA"
filename = "TSLA_stock_data.csv"

# Retry settings for YFinance rate limits
max_retries = 5
retry_delay = 30  # seconds

for attempt in range(max_retries):
    try:
        # Download stock data
        stock_data = yf.download(stock_symbol, start="2020-01-01", end="2024-03-01")
        
        # Check if data is empty
        if stock_data.empty:
            raise ValueError("Downloaded stock data is empty.")
        
        # Save to CSV
        stock_data.to_csv(filename)
        print(f"Stock data saved to {filename}")
        break  # Exit loop if successful

    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Exiting.")
            exit(1)
