import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

class Model(nn.Module):
    """Creates the LSTM model given hyperparams hidden size and number of layers"""
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # keep only last output
        return x
    
def get_train_test(dataset):
    """
    Given a dataset, format, scale, and split the data into train and test sets
    """
    df = pd.read_csv(dataset)
    df['snapped_at'] = pd.to_datetime(df['snapped_at']).dt.strftime('%Y-%m-%d')
    df.set_index('snapped_at', inplace = True)
    df.sort_index(inplace = True)
    df.index = pd.to_datetime(df.index)
    filtered = df[df.index >= pd.to_datetime("2023-01-01")]
    # use 'price' as the target for prediction, and the other features as input
    timeseries = filtered[['price', 'market_cap', 'total_volume']].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    timeseries_scaled = scaler.fit_transform(timeseries)
    print(timeseries_scaled)

    # train-test split for time series
    train_size = int(len(timeseries_scaled) * 0.7)
    #test_size = len(timeseries_scaled) - train_size
    train, test = timeseries_scaled[:train_size], timeseries_scaled[train_size:]
    return train, test, scaler
 
def create_dataset(dataset, lookback):
    """
    Adapted from https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
    Prepares a timeseries dataset for prediction by adding lookback window
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, :] 
        target = dataset[i+lookback, 0] 
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

def train_and_evaluate(hidden_size, lr, lookback, num_layers, batch_size, train, test):
    """
    Given hyperparams and data, return a trained lstm model and its train/test rmse results
    """
    
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
    """
    Perform random hyperparameter search to find what returns the best test rmse value
    """
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

def predict_future(model, scaler, train_data, lookback, num_steps):
    """
    Predict prices for the next num_steps number of days
    """
    # transform data again
    train_scaled = scaler.transform(train_data)
    last_train_input = torch.tensor([train_scaled[-lookback:]], dtype=torch.float32)
    
    model.eval()
    predictions = []

    # Predict for next num_steps number of days
    for _ in range(num_steps):
        with torch.no_grad():
            y_pred = model(last_train_input)
            predictions.append(y_pred.item())
            
            #Remove first timestep and pass onto the next day
            new_input = last_train_input[0, 1:, :].unsqueeze(0)
            new_input = torch.cat((new_input, torch.tensor([[[y_pred.item(), 0.0, 0.0]]], dtype=torch.float32)), dim=1)
            last_train_input = new_input
    
    # These will be in scaled form and need to be inversed to be interpretable
    predictions_scaled = np.array(predictions).reshape(-1, 1)
    predictions_original = scaler.inverse_transform(np.hstack([predictions_scaled, np.zeros((predictions_scaled.shape[0], 2))]))[:,0]
    return predictions_original

def predict_prices(dataset, num_steps):
    train, test, scaler = get_train_test(dataset)
    # best_params = hp_search(train, test, 1)
    # model, test_rmse, train_rmse = train_and_evaluate(best_params['hidden_size'],
    #                                                   best_params['lr'],
    #                                                   best_params['lookback'],
    #                                                   best_params['num_layers'],
    #                                                   best_params['batch_size'],
    #                                                   train,
    #                                                   test)
    model, test_rmse, train_rmse = train_and_evaluate(150, 0.0005, 20, 1, 32, train, test)
    preds = predict_future(model, scaler, train, lookback=20, num_steps=num_steps)
    print(f"Train RMSE: {train_rmse}\n Test RMSE: {test_rmse}")
    # Returning predictions to interpretable values
    print(f"Future predictions for the next {num_steps} days: {preds}")
    return preds

def get_crypto_data(crypto_name):
    CRYPTO_DATASETS = {
        'Bitcoin': 'data/btc-usd-max.csv',
        'Ethereum': 'data/eth-usd-max.csv',
        'Tether': 'data/usdt-usd-max.csv'
    }
    
    return CRYPTO_DATASETS.get(crypto_name, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('crypto_name', type=str, help='Name of the cryptocurrency (e.g. Bitcoin)')
    parser.add_argument('num_days', type=int, help='Number of days ahead to predict')

    args = parser.parse_args()

    dataset = get_crypto_data(args.crypto_name)
    if not dataset:
        return

    predictions = predict_prices(dataset, args.num_days)
    return f"These are the predicted prices for {args.crypto_name} over the next {args.num_days} days: {predictions}"

if __name__ == "__main__":
    main()

