import requests

def get_crypto_prices(crypto_ids):
    """
    Fetches the real-time prices for given crypto IDs.
    :param crypto_ids: List of CoinGecko crypto IDs
    :return: Dictionary {crypto_id: price}
    """
    ids_string = ",".join(crypto_ids)  # Convert list to API format
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        prices = response.json()
        return prices
    
    except requests.exceptions.RequestException as e:
        print("⚠️ Error fetching prices:", e)
        return {}

# Example Usage
crypto_ids = ["bitcoin", "ethereum", "dogecoin"]  # Example cryptos
prices = get_crypto_prices(crypto_ids)

print("\n📊 Current Prices:")
for crypto, data in prices.items():
    print(f"{crypto.capitalize()}: ${data['usd']}")
