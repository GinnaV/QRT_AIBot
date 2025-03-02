# Predicting Cryptocurrency Prices
Timeseries prediction via long short-term memory (LSTM) deep learning! This approach was inspired by this paper: https://thesai.org/Downloads/Volume14No8/Paper_37-Prediction_of_Cryptocurrency_Price_using_Time_Series_Data.pdf


# What prediction.py does
## Training + Inference (Slow)
To train the model on new data and make a prediction, enter the following arguments:
- Name of cryptocurrency (string, case-sensitive)
- Number of days in the future to predict (int)
- Path to CSV file
    - The CSV file must follow the format produced by Coin Gecko API:
    - Column names:
    `snapped_at`, `price`, `market_cap`, `total_volume`
    - Format:
    YYYY-MM-DD 00:00:00 UTC, float, float, float

<b>Warning</b>: training is very slow because it is also running a hyperparameter search. For demo purposes, I would recommend changing the value of `searches` in the call to `hp_search(train, test, searches)` to 1, which will return a random set of values.

<b>Warning</b>: Training will also overwrite any models/saved hyperparameters for existing models of the same name!

## Inference Only (fast)
To make an inference, only enter the following arguments:
- Name of cryptocurrency (string, case-sensitive). The following models are already available:
    - Bitcoin
    - Ethereum
    - Tether
- Number of days in the future to predict (int)

## Output
This model will output a string that can be fed to the chatbot:
"These are the predicted prices for [cryptocurrency] over the next [number] of days: [list of predicted prices]"

# Prediction.ipynb
This includes exploratory analysis of the data and preliminary attempts at training models

# Reddit.ipynb
This file includes calls to Reddit and is customizable to pull relevant posts/comments about select Cryptocurrencies on select subreddits for sentiment analysis.