import requests
import openai
from dotenv import load_dotenv
import os
import re
import datetime

# Load API Key
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of supported cryptocurrencies
crypto_ids = [
    "bitcoin", "ethereum", "dogecoin", "ripple", "cardano", "solana", "polkadot",
    "litecoin", "chainlink", "stellar", "monero", "tron", "avalanche-2", "uniswap", "algorand"
]

crypto_names_map = {name: name for name in crypto_ids}  # Direct mapping

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

def get_crypto_history(crypto_id, days=3):
    """Fetches past 'days' days' historical prices for a given cryptocurrency."""
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert timestamps to readable dates with timezone awareness
        history = {
            datetime.datetime.fromtimestamp(item[0] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d'): item[1]
            for item in data["prices"]
        }
        return history

    except requests.exceptions.RequestException as e:
        print(f" Error fetching historical data for {crypto_id}: {e}")
        return {}

# Fetch prices initially
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
    Processes human-like crypto queries and returns a response with real-time or historical prices.
    """
    crypto_name = extract_crypto_name(user_query)
    response_text = ""

    if crypto_name:
        price_data = crypto_prices.get(crypto_name, {}).get("usd", None)
        history_data = get_crypto_history(crypto_name, days=3)  # Fetch past 3 days' data

        if price_data:
            response_text += f" The current price of {crypto_name.capitalize()} is ${price_data} USD."
        else:
            response_text += f" Sorry, I couldn't find the latest price for {crypto_name.capitalize()}."

        if history_data:
            response_text += "\n Past 3 Days Prices:\n"
            for date, price in history_data.items():
                response_text += f"- {date}: ${price:.2f}\n"

    else:
        response_text = " Sorry, I couldn't find the price for that cryptocurrency. Please try again!"

    # Check for specific queries
    if re.search(r"should i (buy|sell)", user_query, re.IGNORECASE):
        trade_action = "buy" if "buy" in user_query else "sell"
        response_text += f"\n Trading Advice: Based on recent trends, I can provide insights. Would you like a technical analysis on {crypto_name.capitalize()} before making a decision?"
    
    elif re.search(r"(support|resistance|trend|rsi|macd)", user_query, re.IGNORECASE):
        response_text += f"\n Technical Analysis: Would you like a breakdown of support & resistance levels for {crypto_name.capitalize()}?"

    elif re.search(r"news|sentiment", user_query, re.IGNORECASE):
        response_text += f"\n Market Sentiment: I'm working on integrating a real-time news sentiment API for {crypto_name.capitalize()}!"

    # Generate a more natural AI response
    prompt = f"""
    You are a professional cryptocurrency trading assistant. Your goal is to provide real-time and historical prices,
    as well as offer intelligent trading insights based on technical analysis.

    Examples:
    - User: "How much is Bitcoin today?"
      AI: "Bitcoin is currently trading at $42,000 USD. Would you like to know its past trends?"
    - User: "Should I buy Ethereum?"
      AI: "Ethereum has been trending upward in the last few days. Would you like to check the latest indicators before deciding?"

    Now, answer this user query:
    - User: "{user_query}"
    - AI:
    """

    full_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly but professional crypto assistant providing real-time and historical prices, trading insights, and technical analysis."},
            {"role": "user", "content": prompt + response_text}
        ]
    )

    return full_response.choices[0].message.content

# Take user input dynamically
while True:
    user_query = input("\n Enter your crypto query (or type 'exit' to quit): ").strip()
    
    if user_query.lower() == "exit":
        print(" Exiting... Have a great day!")
        break
    
    print(f"\n AI: {ask_crypto_ai(user_query)}")