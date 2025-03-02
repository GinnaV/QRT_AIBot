import requests
import pandas as pd
import datetime

def generate_crypto_csv(crypto_id="bitcoin"):
    """
    Fetch the last 7 days of historical price data from CoinGecko 
    and generate a new CSV file with a timestamp.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days=7&interval=daily"
    
    try:
        response = requests.get(url).json()
        prices = [entry[1] for entry in response["prices"]]  # Extract only prices
        
        # Create a DataFrame
        df = pd.DataFrame({"Day": range(1, 8), "Price": prices})
        
        # Generate a filename with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{crypto_id}_{timestamp}.csv"
        
        # Save the CSV file
        df.to_csv(filename, index=False)
        
        print(f" New CSV file generated: {filename}")
        return filename
    
    except requests.exceptions.RequestException as e:
        print(f" Error fetching data for {crypto_id}: {e}")
        return None

# Example Usage
crypto_name = input("Enter a cryptocurrency (e.g., bitcoin, ethereum, solana): ").lower()
csv_file = generate_crypto_csv(crypto_name)
