import requests
import pandas as pd
import datetime
import os

def generate_crypto_csv(crypto_id="bitcoin"):
    """Fetch the last 7 days of historical price data from CoinGecko and generate a CSV file."""
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days=7&interval=daily"
    
    try:
        response = requests.get(url).json()
        
        # Extract prices safely
        prices = [entry[1] for entry in response.get("prices", [])]

        # Debugging: Print number of fetched price points
        print(f"Fetched {len(prices)} price points for {crypto_id}")

        # Ensure there are exactly 7 price points
        if len(prices) < 7:
            print(f" Warning: Expected 7 days of data, but received {len(prices)}. Padding missing values.")
            while len(prices) < 7:
                prices.append(prices[-1] if prices else 0.0)  # Repeat last known price if missing

        elif len(prices) > 7:
            print(f" Warning: Received more than 7 price points, trimming to 7.")
            prices = prices[:7]

        # Ensure consistency
        days = list(range(1, len(prices) + 1))
        
        df = pd.DataFrame({"Day": days, "Price": prices})

        filename = f"data/{crypto_id}-latest.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(filename, index=False)

        print(f" New CSV file generated: {filename}")
        return filename
    
    except requests.exceptions.RequestException as e:
        print(f" Error fetching data for {crypto_id}: {e}")
        return None
