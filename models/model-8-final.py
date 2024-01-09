import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StockPricePredictor(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        learning_rate: float,
        train_losses: list[int],
        test_losses: list[int],
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        self.train_losses.append(self.trainer.callback_metrics["train_loss"].item())
        self.test_losses.append(self.trainer.callback_metrics["val_loss"].item())


def run(w: int):
    settings = {
        "train_size": 0.80,
        "subsample": False,
        "subsample_size": 0.2,
        "stock_tickers": [
            "META",
            "GOOGL",
            "NFLX",
            "AMZN",
            "TSLA",
            "HD",
            "PG",
            "COST",
            "PEP",
            "XOM",
            "CVX",
            "COP",
            "MA",
            "JPM",
            "V",
            "UNH",
            "JNJ",
            "LLY",
            "UNP",
            "BA",
            "CAT",
            "LIN",
            "SHW",
            "FCX",
            "PLD",
            "AMT",
            "EQIX",
            "MSFT",
            "AAPL",
            "AVGO",
            "NEE",
            "SO",
            "DUK",
        ],
        "lookback_window": w,
        "lookahead_window": 1,
        "input_size": 2,
        "num_layers": 4,
        "hidden_size": 32,
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "shuffle_train": False,
        "accelerator": "gpu",
    }

    with open("../data/final/sp500-price-volume.pkl", "rb") as f:
        sp500_df = pkl.load(f)

    with open("../data/final/sp500-price-volume-sentiment.pkl", "rb") as f:
        sp500_sentiment_df = pkl.load(f)

    sp500_df["Ticker"] = sp500_df["Ticker"].astype("category")
    sp500_df["Date"] = sp500_df["Date"].astype("datetime64[ns]")
    sp500_df["Price"] = pd.to_numeric(
        sp500_df["Price"], errors="coerce", downcast="float"
    )
    sp500_df["Volume"] = pd.to_numeric(
        sp500_df["Volume"], errors="coerce", downcast="integer"
    )

    sp500_sentiment_df["Ticker"] = sp500_sentiment_df["Ticker"].astype("category")
    sp500_sentiment_df["Date"] = sp500_sentiment_df["Date"].astype("datetime64[ns]")
    sp500_sentiment_df["Sentiment"] = pd.to_numeric(
        sp500_sentiment_df["Sentiment"], errors="coerce", downcast="float"
    )
    sp500_sentiment_df["Price"] = pd.to_numeric(
        sp500_sentiment_df["Price"], errors="coerce", downcast="float"
    )
    sp500_sentiment_df["Volume"] = pd.to_numeric(
        sp500_sentiment_df["Volume"], errors="coerce", downcast="integer"
    )

    def split_data(
        df: pd.DataFrame, train_size: float = 0.8, lbp: int = 30, lfp: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        lbp: look back period
        lfp: look forward period
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        tickers = df["Ticker"].unique()

        for ticker in tickers:
            ticker_df = df[df["Ticker"] == ticker]
            ticker_df = ticker_df.drop("Ticker", axis=1)
            ticker_df = ticker_df.set_index("Date")
            ticker_array = ticker_df.sort_index().values

            train_count = int(ticker_array.shape[0] * train_size)
            train = ticker_array[:train_count]
            test = ticker_array[train_count:]

            ticker_X_train = []
            ticker_y_train = []
            ticker_X_test = []
            ticker_y_test = []

            for i in range(lbp, train.shape[0] - lfp + 1):
                ticker_X_train.append(train[i - lbp : i, :])
                ticker_y_train.append(train[i : i + lfp, 0])

            for i in range(lbp, test.shape[0] - lfp + 1):
                ticker_X_test.append(test[i - lbp : i, :])
                ticker_y_test.append(test[i : i + lfp, 0])

            X_train.append(ticker_X_train)
            y_train.append(ticker_y_train)
            X_test.append(ticker_X_test)
            y_test.append(ticker_y_test)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))

        X_train = X_scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2], -1)
        X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
            X_test.shape[0] * X_test.shape[1], X_test.shape[2], -1
        )
        y_train = y_scaler.fit_transform(
            y_train.reshape(-1, y_train.shape[-1])
        ).reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2])
        y_test = y_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(
            y_test.shape[0] * y_test.shape[1], y_test.shape[2]
        )

        return X_train, y_train, X_test, y_test, y_scaler

    (
        X_train_no_sentiment,
        y_train_no_sentiment,
        X_test_no_sentiment,
        y_test_no_sentiment,
        y_scaler_no_sentiment,
    ) = split_data(
        sp500_df,
        train_size=settings["train_size"],
        lbp=settings["lookback_window"],
        lfp=settings["lookahead_window"],
    )
    (
        X_train_sentiment,
        y_train_sentiment,
        X_test_sentiment,
        y_test_sentiment,
        y_scaler_sentiment,
    ) = split_data(
        sp500_sentiment_df,
        train_size=settings["train_size"],
        lbp=settings["lookback_window"],
        lfp=settings["lookahead_window"],
    )

    print(f"X_train_no_sentiment shape: {X_train_no_sentiment.shape}")
    print(f"y_train_no_sentiment shape: {y_train_no_sentiment.shape}")
    print(f"X_test_no_sentiment shape: {X_test_no_sentiment.shape}")
    print(f"y_test_no_sentiment shape: {y_test_no_sentiment.shape}")
    print()
    print(f"X_train_sentiment shape: {X_train_sentiment.shape}")
    print(f"y_train_sentiment shape: {y_train_sentiment.shape}")
    print(f"X_test_sentiment shape: {X_test_sentiment.shape}")
    print(f"y_test_sentiment shape: {y_test_sentiment.shape}")

    no_sentiment_train_dataset = TensorDataset(
        torch.from_numpy(X_train_no_sentiment).float(),
        torch.from_numpy(y_train_no_sentiment).float(),
    )
    no_sentiment_test_dataset = TensorDataset(
        torch.from_numpy(X_test_no_sentiment).float(),
        torch.from_numpy(y_test_no_sentiment).float(),
    )

    sentiment_train_dataset = TensorDataset(
        torch.from_numpy(X_train_sentiment).float(),
        torch.from_numpy(y_train_sentiment).float(),
    )
    sentiment_test_dataset = TensorDataset(
        torch.from_numpy(X_test_sentiment).float(),
        torch.from_numpy(y_test_sentiment).float(),
    )

    no_sentiment_train_dataloader = DataLoader(
        no_sentiment_train_dataset,
        batch_size=settings["batch_size"],
        shuffle=settings["shuffle_train"],
        # num_workers=11,
        # persistent_workers=True,
    )
    no_sentiment_test_dataloader = DataLoader(
        no_sentiment_test_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        # num_workers=11,
        # persistent_workers=True,
    )

    sentiment_train_dataloader = DataLoader(
        sentiment_train_dataset,
        batch_size=settings["batch_size"],
        shuffle=settings["shuffle_train"],
        # num_workers=11,
        # persistent_workers=True,
    )
    sentiment_test_dataloader = DataLoader(
        sentiment_test_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        # num_workers=11,
        # persistent_workers=True,
    )

    no_sentiment_train_losses = []
    no_sentiment_test_losses = []
    sentiment_train_losses = []
    sentiment_test_losses = []

    no_sentiment_model = StockPricePredictor(
        2,
        settings["hidden_size"],
        settings["num_layers"],
        settings["lookahead_window"],
        learning_rate=settings["learning_rate"],
        train_losses=no_sentiment_train_losses,
        test_losses=no_sentiment_test_losses,
    )
    sentiment_model = StockPricePredictor(
        3,
        settings["hidden_size"],
        settings["num_layers"],
        settings["lookahead_window"],
        learning_rate=settings["learning_rate"],
        train_losses=sentiment_train_losses,
        test_losses=sentiment_test_losses,
    )

    no_sentiment_trainer = L.Trainer(
        max_epochs=settings["max_epochs"], accelerator=settings["accelerator"]
    )
    no_sentiment_trainer.fit(
        no_sentiment_model, no_sentiment_train_dataloader, no_sentiment_test_dataloader
    )

    sentiment_trainer = L.Trainer(
        max_epochs=settings["max_epochs"], accelerator=settings["accelerator"]
    )
    sentiment_trainer.fit(
        sentiment_model, sentiment_train_dataloader, sentiment_test_dataloader
    )

    no_sentiment_train_losses = np.array(no_sentiment_train_losses).reshape(
        -1, settings["max_epochs"]
    )
    no_sentiment_test_losses = np.array(no_sentiment_test_losses).reshape(
        -1, settings["max_epochs"]
    )
    sentiment_train_losses = np.array(sentiment_train_losses).reshape(
        -1, settings["max_epochs"]
    )
    sentiment_test_losses = np.array(sentiment_test_losses).reshape(
        -1, settings["max_epochs"]
    )

    print(f"Done training lookback window {settings['lookback_window']}")

    print(f"No Sentiment Final RMSE: {no_sentiment_test_losses.mean(axis=0)[-1]}")
    print(f"With Sentiment Final RMSE: {sentiment_test_losses.mean(axis=0)[-1]}")

    no_sentiment_model.eval()
    sentiment_model.eval()

    y_hat_no_sentiment = []
    y_hat_sentiment = []

    for sample in range(X_test_sentiment.shape[0]):
        ns_X = (
            torch.from_numpy(X_test_no_sentiment[sample, :, :])
            .float()
            .reshape(1, settings["lookback_window"], -1)
        )
        s_X = (
            torch.from_numpy(X_test_sentiment[sample, :, :])
            .float()
            .reshape(1, settings["lookback_window"], -1)
        )

        y_hat_no_sentiment.append(no_sentiment_model(ns_X)[0, 0].item())
        y_hat_sentiment.append(sentiment_model(s_X)[0, 0].item())

    y_hat_no_sentiment = np.array(y_hat_no_sentiment)
    y_hat_sentiment = np.array(y_hat_sentiment)

    with open(
        f"./final/lookback-{settings['lookback_window']}/no-sentiment-model.pkl", "wb"
    ) as f:
        pkl.dump(no_sentiment_model, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/sentiment-model.pkl", "wb"
    ) as f:
        pkl.dump(sentiment_model, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/no-sentiment-scaler.pkl", "wb"
    ) as f:
        pkl.dump(y_scaler_no_sentiment, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/sentiment-scaler.pkl", "wb"
    ) as f:
        pkl.dump(y_scaler_sentiment, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/no-sentiment-train-losses.pkl",
        "wb",
    ) as f:
        pkl.dump(no_sentiment_train_losses, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/no-sentiment-test-losses.pkl",
        "wb",
    ) as f:
        pkl.dump(no_sentiment_test_losses, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/sentiment-train-losses.pkl",
        "wb",
    ) as f:
        pkl.dump(sentiment_train_losses, f)

    with open(
        f"./final/lookback-{settings['lookback_window']}/sentiment-test-losses.pkl",
        "wb",
    ) as f:
        pkl.dump(sentiment_test_losses, f)

    print(f"Done with lookback window {settings['lookback_window']}")


if __name__ == "__main__":
    for r in [15, 30, 60, 90]:
        run(r)
