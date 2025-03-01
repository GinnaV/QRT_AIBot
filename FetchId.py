import requests

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
        print("⚠️ Error fetching crypto list from CoinGecko:", e)
        return {}  # Return an empty dictionary if an error occurs

# Example Usage
crypto_dict = get_crypto_ids()

if crypto_dict:
    print("\n✅ Total Cryptocurrencies Available:", len(crypto_dict))
    print("\n🔹 First 10 Cryptos Available:", list(crypto_dict.items())[:10])  # Show first 10 cryptos
else:
    print("\n⚠️ No data retrieved. Please check your API connection.")
