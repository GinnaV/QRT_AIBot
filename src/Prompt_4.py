import openai
import os
import re
import sys
import json
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from prediction import predict_prices, get_crypto_model  # Import prediction functions

# Load API Key
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of supported cryptocurrencies (only those with stored data)
crypto_ids = [
    "bitcoin", "ethereum", "tether"
]

crypto_names_map = {name: name for name in crypto_ids}  # Direct mapping

# Dictionary to store chat memory (user's selected crypto)
chat_memory = {"selected_crypto": None, "history": []}

def extract_crypto_name(query):
    """Extracts cryptocurrency name from the user query."""
    query = query.lower()
    for name in crypto_names_map.keys():
        if re.search(rf"\b{name}\b", query):  # Match whole word
            return crypto_names_map[name]
    return None

def ask_crypto_ai(user_query):
    """Handles user queries with stored data predictions only."""
    global chat_memory

    crypto_name = extract_crypto_name(user_query)

    if crypto_name:
        chat_memory["selected_crypto"] = crypto_name
    else:
        crypto_name = chat_memory["selected_crypto"]

    if not crypto_name:
        return " Please specify a cryptocurrency to continue (e.g., 'Tell me about Bitcoin')."

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
        return f" Predicted prices for {crypto_name.capitalize()} over the next {num_days} days:\n{prediction_text}"

    return f" I'm tracking {crypto_name.capitalize()}. What would you like to know? Type 'predict 5 days' to get a future price estimate."

# Continuous Chat
while True:
    print("\n Enter your crypto query (or type 'exit' to quit): ", end="", flush=True) 
    user_query = sys.stdin.readline().strip()

    if user_query.lower() == "exit":
        print(" Exiting... Have a great day!")
        break
    
    print(f"\n AI: {ask_crypto_ai(user_query)}")
