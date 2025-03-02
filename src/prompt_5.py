import requests
import openai
from dotenv import load_dotenv
import os
import re
import datetime
import sys
import torch
import pandas as pd
import numpy as np
from prediction import predict_prices, get_crypto_model  # Import prediction functions

# Load API Key
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of supported cryptocurrencies
crypto_ids = [
    "bitcoin", "ethereum", "dogecoin", "ripple", "cardano", "solana", "polkadot",
    "litecoin", "chainlink", "stellar", "monero", "tron", "avalanche-2", "uniswap", "algorand"
]

crypto_names_map = {name: name for name in crypto_ids}  # Direct mapping

# Dictionary to store chat memory (user's selected crypto)
chat_memory = {"selected_crypto": None, "history": []}

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
        
        # Convert timestamps to readable dates
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
    Handles user queries with memory of past conversations and integrates predictions.
    """
    global chat_memory

    # Extract cryptocurrency name
    crypto_name = extract_crypto_name(user_query)

    # If user mentions a new crypto, update memory
    if crypto_name:
        chat_memory["selected_crypto"] = crypto_name
    else:
        # If user doesn't mention one, use the last selected crypto
        crypto_name = chat_memory["selected_crypto"]

    if not crypto_name:
        return " Please specify a cryptocurrency to continue (e.g., 'Tell me about Bitcoin')."

    response_text = f"Talking about {crypto_name.capitalize()}:\n"

    # Fetch price and historical data
    price_data = crypto_prices.get(crypto_name, {}).get("usd", None)
    history_data = get_crypto_history(crypto_name, days=3)

    if price_data:
        response_text += f" The current price of {crypto_name.capitalize()} is *${price_data} USD*."
    else:
        response_text += f" Sorry, I couldn't find the latest price for {crypto_name.capitalize()}."

    if history_data:
        response_text += "\n *Past 3 Days Prices:*\n"
        for date, price in history_data.items():
            response_text += f"- {date}: *${price:.2f}*\n"

    # Handle price prediction request
    match = re.search(r'predict (\d+) days?', user_query, re.IGNORECASE)
    if match:
        num_days = int(match.group(1))

        # Load trained model for the requested cryptocurrency
        model, best_params = get_crypto_model(crypto_name)
        if model is None:
            return f" Sorry, no trained model is available for {crypto_name.capitalize()}."

        # Predict prices using stored data
        predictions = predict_prices(model, crypto_name, best_params['lookback'], num_days)

        prediction_text = "\n".join([f"Day {i+1}: ${pred:.2f}" for i, pred in enumerate(predictions)])
        return f"  Predicted prices for {crypto_name.capitalize()} over the next {num_days} days:\n{prediction_text}"

    # Check for trading advice, technical analysis, or sentiment analysis
    if re.search(r"should i (buy|sell)", user_query, re.IGNORECASE):
        response_text += f"\n *Trading Advice:* Want me to analyze trends for {crypto_name.capitalize()} before deciding?"

    elif re.search(r"(support|resistance|trend|rsi|macd)", user_query, re.IGNORECASE):
        response_text += f"\n *Technical Analysis:* Would you like a breakdown of support & resistance levels for {crypto_name.capitalize()}?"

    elif re.search(r"news|sentiment", user_query, re.IGNORECASE):
        response_text += f"\n *Market Sentiment:* I'm working on integrating a real-time news sentiment API for {crypto_name.capitalize()}!"

    # Add the current query to chat memory
    chat_memory["history"].append({"role": "user", "content": user_query})
    chat_memory["history"].append({"role": "assistant", "content": response_text})

    # Generate AI response with context
    prompt = f"""
    You are a cryptocurrency trading assistant.
    You remember the user's chosen cryptocurrency and keep conversation context.
    The current selected cryptocurrency is {crypto_name.capitalize()}.
    
    User Query: "{user_query}"
    AI Response:
    """

    full_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a crypto assistant who remembers past conversations and keeps track of the selected cryptocurrency."}
        ] + chat_memory["history"]
    )

    return full_response.choices[0].message.content

# Continuous Chat
while True:
    print("\n Enter your crypto query (or type 'exit' to quit): ", end="", flush=True) 
    user_query = sys.stdin.readline().strip()

    if user_query.lower() == "exit":
        print(" Exiting... Have a great day!")
        break
    
    print(f"\n AI: {ask_crypto_ai(user_query)}")
