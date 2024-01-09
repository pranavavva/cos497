import numpy as np
import pandas as pd
import pickle as pkl
import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import trange


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


with open("../data/final/sp500-price-volume.pkl", "rb") as f:
    sp500_df = pkl.load(f)

with open("../data/final/sp500-price-volume-sentiment.pkl", "rb") as f:
    sp500_sentiment_df = pkl.load(f)


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

    X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape[0] * X_train.shape[1], X_train.shape[2], -1
    )
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape[0] * X_test.shape[1], X_test.shape[2], -1
    )
    y_train = y_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(
        y_train.shape[0] * y_train.shape[1], y_train.shape[2]
    )
    y_test = y_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(
        y_test.shape[0] * y_test.shape[1], y_test.shape[2]
    )

    return X_train, y_train, X_test, y_test, y_scaler


ns_means = []
ns_stds = []
s_means = []
s_stds = []
p_values = []
all_y_hat_ns = []
all_y_hat_s = []
all_y_test_ns = []
all_y_test_s = []

for lookback in [15, 30, 60, 90]:
    print("==================================================")
    print(f"lookback: {lookback}")
    print("==================================================")
    with open(f"./final/lookback-{lookback}/no-sentiment-model.pkl", "rb") as f:
        ns_model = pkl.load(f)

    with open(f"./final/lookback-{lookback}/sentiment-model.pkl", "rb") as f:
        s_model = pkl.load(f)

    with open(f"./final/lookback-{lookback}/no-sentiment-test-losses.pkl", "rb") as f:
        ns_test_losses = pkl.load(f).reshape(-1)

    with open(f"./final/lookback-{lookback}/sentiment-test-losses.pkl", "rb") as f:
        s_test_losses = pkl.load(f).reshape(-1)

    (_, _, X_test_ns, y_test_ns, _) = split_data(
        sp500_df, train_size=0.8, lbp=lookback, lfp=1
    )
    (_, _, X_test_s, y_test_s, _) = split_data(
        sp500_sentiment_df, train_size=0.8, lbp=lookback, lfp=1
    )

    print(f"X_test_ns shape: {X_test_ns.shape}")
    print(f"y_test_ns shape: {y_test_ns.shape}")
    print()
    print(f"X_test_s shape: {X_test_s.shape}")
    print(f"y_test_s shape: {y_test_s.shape}")

    print(f"Final No Sentiment Test Loss: {ns_test_losses[-1]}")
    print(f"Final Sentiment Test Loss: {s_test_losses[-1]}")

    ns_model.eval()
    s_model.eval()

    y_hat_ns = []
    y_hat_s = []

    with torch.no_grad():
        for i in trange(X_test_ns.shape[0]):
            ns_X = torch.from_numpy(X_test_ns[i, :, :]).float().reshape(1, lookback, -1)
            s_X = torch.from_numpy(X_test_s[i, :, :]).float().reshape(1, lookback, -1)

            y_hat_ns.append(ns_model(ns_X)[0, 0].item())
            y_hat_s.append(s_model(s_X)[0, 0].item())

    y_hat_ns = np.array(y_hat_ns)
    y_hat_s = np.array(y_hat_s)
    y_test = y_test_ns[:, 0]

    y_hat_s_error = np.sqrt((y_hat_s - y_test) ** 2)
    y_hat_ns_error = np.sqrt((y_hat_ns - y_test) ** 2)
    y_hat_diff = y_hat_s_error - y_hat_ns_error

    print(f"No Sentiment Mean Error: {y_hat_ns_error.mean():.4f}")
    print(f"No Sentiment Std Error: {y_hat_ns_error.std():.4f}")
    print()
    print(f"With Sentiment Mean Error: {y_hat_s_error.mean():.4f}")
    print(f"With Sentiment Std Error: {y_hat_s_error.std():.4f}")

    # determine if the difference is statistically significant using wilcoxon signed-rank test

    from scipy.stats import ttest_rel

    stat, p = ttest_rel(y_hat_s_error, y_hat_ns_error, alternative="less")
    print(f"stat: {stat}, p-value: {p}")

    if p > 0.05:
        print("y_hat_s_error not significantly less than y_hat_ns_error")
    else:
        print("y_hat_s_error significantly less than y_hat_ns_error")

    ns_means.append(y_hat_ns_error.mean())
    ns_stds.append(y_hat_ns_error.std())
    s_means.append(y_hat_s_error.mean())
    s_stds.append(y_hat_s_error.std())
    p_values.append(p)
    all_y_hat_ns.append(y_hat_ns)
    all_y_hat_s.append(y_hat_s)
    all_y_test_ns.append(y_test_ns)
    all_y_test_s.append(y_test_s)

df_stats = pd.DataFrame(
    {
        "lookback": [15, 30, 60, 90],
        "ns_means": ns_means,
        "ns_stds": ns_stds,
        "s_means": s_means,
        "s_stds": s_stds,
        "p_values": p_values,
    }
)

with open("./final/all-y-hat-ns.pkl", "wb") as f:
    pkl.dump(all_y_hat_ns, f)

with open("./final/all-y-hat-s.pkl", "wb") as f:
    pkl.dump(all_y_hat_s, f)

with open("./final/all-y-test-ns.pkl", "wb") as f:
    pkl.dump(all_y_test_ns, f)

with open("./final/all-y-test-s.pkl", "wb") as f:
    pkl.dump(all_y_test_s, f)

df_stats.to_csv("./final/stats.csv", index=False)
print(df_stats)
