import argparse
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import os
import requests
import datetime

class Model(nn.Module):
    """Creates the LSTM model given hyperparameters hidden size and number of layers"""
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # keep only last output
        return x

def get_train_test(dataset):
    """Given a dataset, format, scale, and split the data into train and test sets"""
    df = pd.read_csv(dataset)
    df['snapped_at'] = pd.to_datetime(df['snapped_at']).dt.strftime('%Y-%m-%d')
    df.set_index('snapped_at', inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    filtered = df[df.index >= pd.to_datetime("2023-01-01")]
    timeseries = filtered[['price', 'market_cap', 'total_volume']].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    timeseries_scaled = scaler.fit_transform(timeseries)

    train_size = int(len(timeseries_scaled) * 0.7)
    train, test = timeseries_scaled[:train_size], timeseries_scaled[train_size:]
    return train, test, scaler

def create_dataset(dataset, lookback):
    """Prepares a timeseries dataset for prediction by adding lookback window"""
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :]
        target = dataset[i+lookback, 0]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

def train_and_evaluate(hidden_size, lr, lookback, num_layers, batch_size, train, test):
    """Given hyperparams and data, return a trained lstm model and its train/test rmse results"""
    X_train, y_train = create_dataset(train, lookback)
    X_test, y_test = create_dataset(test, lookback)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)

    model = Model(hidden_size=hidden_size, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_train_pred, y_train).item())
        y_test_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_test, y_test_pred))

    return model, test_rmse, train_rmse

def hp_search(train, test, searches):
    """Perform random hyperparameter search to find what returns the best test rmse value"""
    param_space = {
        'hidden_size': [20, 50, 100, 150],
        'lr': [0.0001, 0.0005, 0.001],
        'lookback': [5, 10, 15, 20],
        'num_layers': [1, 2],
        'batch_size': [8, 16, 32]
    }

    best_rmse = float('inf')
    best_params = {}

    for i in range(searches):
        hidden_size = random.choice(param_space['hidden_size'])
        lr = random.choice(param_space['lr'])
        lookback = random.choice(param_space['lookback'])
        num_layers = random.choice(param_space['num_layers'])
        batch_size = random.choice(param_space['batch_size'])

        print(f"Trying hidden_size={hidden_size}, lr={lr}, lookback={lookback}, num_layers={num_layers}, batch_size={batch_size}")

        _, test_rmse, _ = train_and_evaluate(hidden_size, lr, lookback, num_layers, batch_size, train, test)

        print(f"Test RMSE: {test_rmse}")

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_params = {
                'hidden_size': hidden_size,
                'lr': lr,
                'lookback': lookback,
                'num_layers': num_layers,
                'batch_size': batch_size
            }

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Test RMSE: {best_rmse}")
    return best_params

def generate_crypto_csv(crypto_id="bitcoin"):
    """Fetch the last 7 days of historical price data from CoinGecko and generate a new CSV file with a timestamp."""
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days=7&interval=daily"
    try:
        response = requests.get(url).json()
        prices = [entry[1] for entry in response["prices"]]
        df = pd.DataFrame({"Day": range(1, 8), "Price": prices})
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"data/{crypto_id}_{timestamp}.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"New CSV file generated: {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {crypto_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('crypto_name', type=str, help='Name of the cryptocurrency (e.g. Bitcoin)')
    parser.add_argument('num_days', type=int, help='Number of days ahead to predict')
    parser.add_argument('new_csv', type=str, nargs='?', default=None, help='If not None, the model will train on the given csv')

    args = parser.parse_args()

    if args.new_csv:
        model, best_params = train_new(args.new_csv, args.crypto_name)
    else:
        model, best_params = get_crypto_model(args.crypto_name)

    dataset = generate_crypto_csv(args.crypto_name)
    if not dataset:
        return

    predictions = predict_prices(model, args.crypto_name, best_params['lookback'], args.num_days)
    output = f"These are the predicted prices for {args.crypto_name} over the next {args.num_days} days: {predictions}"
    print(output)
    return output

if __name__ == "__main__":
    main()
