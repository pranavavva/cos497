# Models

Contains models for this project. There are two types of models: `base` which does not take into account the sentiment data and `sentiment` that does. The `base` models solely predict stock price movement based on historical price and technicals. The `sentiment` models also take into account online sentiment data from Reddit posts and comments. We're only looking at stock price movement from January 1, 2021 to December 31, 2021.

`base-model-1`: Predicts price for $AAPL from 10/2021 to 10/2023 using data from 10/2013 to 10/2021 with sliding window data selection approach with one-shot next-range predicition scheme. Uses LSTM-based model.