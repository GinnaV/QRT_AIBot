import requests
import datetime

def get_crypto_ids():
    """
    Fetches a list of all available cryptocurrencies from CoinGecko 
    and returns a dictionary mapping names to their API-friendly IDs.
    """
    url = "https://api.coingecko.com/api/v3/coins/list"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for HTTP issues (e.g., 404, 500)
        crypto_list = response.json()
        
        # Convert the list into a dictionary: {"bitcoin": "bitcoin", "ethereum": "ethereum", ...}
        crypto_dict = {crypto["name"].lower(): crypto["id"] for crypto in crypto_list}
        return crypto_dict
    
    except requests.exceptions.RequestException as e:
        print("Error fetching crypto list from CoinGecko:", e)
        return {}  # Return an empty dictionary if an error occurs


def get_crypto_history(crypto_id, days=7):
    """
    Fetches the historical price data for a given cryptocurrency over the last 'days' days.
    :param crypto_id: CoinGecko cryptocurrency ID (e.g., 'bitcoin')
    :param days: Number of past days to fetch data for
    :return: List of tuples (timestamp, price)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extracting timestamps and prices using timezone-aware datetime
        prices = [
            (datetime.datetime.fromtimestamp(item[0] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d'), item[1]) 
            for item in data['prices']
        ]
        return prices
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {crypto_id}: {e}")
        return []


# Fetch the available cryptocurrency IDs
crypto_dict = get_crypto_ids()

if crypto_dict:
    print("\nTotal Cryptocurrencies Available:", len(crypto_dict))
    print("\nFirst 10 Cryptos Available:", list(crypto_dict.items())[:10])  # Show first 10 cryptos

    # List of top 10 cryptocurrencies (as of recent trends)
    top_10_crypto_ids = [
        "bitcoin", "ethereum", "binancecoin", "ripple", "solana",
        "cardano", "dogecoin", "tron", "polkadot", "litecoin"
    ]

    # Fetch and display historical prices
    for crypto in top_10_crypto_ids:
        print(f"\nLast 7 Days Prices for {crypto.capitalize()}:")
        history = get_crypto_history(crypto, days=7)
        
        if history:
            for date, price in history:
                print(f"{date}: ${price:.2f}")
        else:
            print(f"No data available for {crypto}.")
else:
    print("\nNo crypto IDs retrieved. Please check your API connection.")
