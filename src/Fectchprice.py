import requests
import datetime

def get_crypto_history(crypto_id, days=3):
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
        print(f" Error fetching data for {crypto_id}: {e}")
        return []

# List of top 10 cryptocurrencies (as of recent trends)
top_10_crypto_ids = [
    "bitcoin", "ethereum", "binancecoin", "ripple", "solana",
    "cardano", "dogecoin", "tron", "polkadot", "litecoin"
]

# Fetch and display historical prices
for crypto in top_10_crypto_ids:
    print(f"\n Last 7 Days Prices for {crypto.capitalize()}:")
    history = get_crypto_history(crypto, days=7)
    
    for date, price in history:
        print(f"{date}: ${price:.2f}")
