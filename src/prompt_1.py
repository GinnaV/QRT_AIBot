import requests
import openai
from dotenv import load_dotenv
import os
import re

# Load API Key
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of supported cryptocurrencies
crypto_ids = [
    "bitcoin", "ethereum", "dogecoin", "ripple", "cardano", "solana", "polkadot",
    "litecoin", "chainlink", "stellar", "monero", "tron", "avalanche-2", "uniswap", "algorand"
]

crypto_names_map = {  # For better matching (ignoring hyphens, capitalization)
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "dogecoin": "dogecoin",
    "ripple": "ripple",
    "cardano": "cardano",
    "solana": "solana",
    "polkadot": "polkadot",
    "litecoin": "litecoin",
    "chainlink": "chainlink",
    "stellar": "stellar",
    "monero": "monero",
    "tron": "tron",
    "avalanche": "avalanche-2",
    "uniswap": "uniswap",
    "algorand": "algorand"
}

def get_crypto_prices():
    """Fetches real-time prices for predefined crypto IDs."""
    ids_string = ",".join(crypto_ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(" Error fetching prices:", e)
        return {}

# Fetch prices initially (can be refreshed periodically)
crypto_prices = get_crypto_prices()

def extract_crypto_name(query):
    """Extracts cryptocurrency name from the user query."""
    query = query.lower()
    for name in crypto_names_map.keys():
        if re.search(rf"\b{name}\b", query):  # Match whole word
            return crypto_names_map[name]
    return None

def ask_crypto_ai(user_query):
    """
    Processes human-like crypto queries and returns a response with real-time prices.
    """
    crypto_name = extract_crypto_name(user_query)

    if crypto_name and crypto_name in crypto_prices:
        price = crypto_prices[crypto_name]["usd"]
        response_text = f"The current price of {crypto_name.capitalize()} is **${price} USD**."
    else:
        response_text = "I'm sorry, I couldn't find the price for that cryptocurrency. Please try again!"

    # Generate a more human-like response using AI
    prompt = f"""
    You are a helpful cryptocurrency assistant. When users ask about crypto prices, respond in a friendly way.

    Example:
    - User: "How much is Bitcoin today?"
    - AI: "Bitcoin is currently trading at **$42,000 USD**. Would you like to know more details?"

    Now, answer this user query:
    - User: "{user_query}"
    - AI:
    """

    full_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a friendly crypto assistant providing real-time prices."},
                  {"role": "user", "content": prompt + response_text}]
    )

    return full_response.choices[0].message.content

# Example Queries
user_queries = [
    "Hey, what's the latest price of Ethereum?",
    "How much is Dogecoin right now?",
    "Is Solana doing well today?",
    "Give me the latest update on Bitcoin!"
]

for query in user_queries:
    print(f"\n User: {query}")
    print(f" AI: {ask_crypto_ai(query)}")
