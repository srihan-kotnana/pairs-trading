import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from itertools import combinations
from typing import List, Tuple, Dict

class PairsTradingEngine:
    def __init__(self, data_dir: str, lookback: int = 60):

        self.data_dir = data_dir
        self.lookback = lookback
        self.prices = self._load_data()
        self.cointegrated_pairs = []

    def _load_data(self) -> pd.DataFrame:
        data = {}
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                ticker = file.replace(".csv", "")
                path = os.path.join(self.data_dir, file)

                try:
                    df = pd.read_csv(path, header=0, skiprows=[1])
                    df.columns = df.columns.str.strip()

                    if "timestamp" not in df.columns:
                        if "Date" in df.columns:
                            df.rename(columns={"Date": "timestamp"}, inplace=True)
                        else:
                            raise ValueError("Missing 'timestamp' column")

                    if "Close" not in df.columns:
                        raise ValueError("Missing 'Close' column")

                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df[["timestamp", "Close"]].rename(columns={"Close": ticker})
                    df.set_index("timestamp", inplace=True)
                    data[ticker] = df

                except Exception as e:
                    print(f"Skipping {ticker}: {e}")

        if not data:
            raise ValueError("No valid CSVs found in data directory.")

        combined = pd.concat(data.values(), axis=1, join="outer")
        combined.sort_index(inplace=True)
        combined.ffill(inplace=True)
        combined.dropna(axis=1, inplace=True)
        return combined





    def find_cointegrated_pairs(self, tickers: List[str] = None, pvalue_threshold: float = 0.05) -> List[Tuple[str, str]]:

        if tickers is None:
            tickers = list(self.prices.columns)
        pairs = combinations(tickers, 2)
        result = []

        for x, y in pairs:
            px = self.prices[x].dropna()
            py = self.prices[y].dropna()
            df = pd.concat([px, py], axis=1).dropna()
            if len(df) < self.lookback * 2:
                continue

            X = sm.add_constant(df[x])
            model = OLS(df[y], X).fit()
            residuals = df[y] - model.predict(X)

            adf_pval = adfuller(residuals)[1]
            if adf_pval < pvalue_threshold:
                result.append((x, y))

        self.cointegrated_pairs = result
        return result

    def calculate_spread_and_zscore(self, x: str, y: str) -> pd.DataFrame:

        df = self.prices[[x, y]].dropna()
        X = sm.add_constant(df[x])
        model = OLS(df[y], X).fit()
        hedge_ratio = model.params[x]
        spread = df[y] - hedge_ratio * df[x]

        zscore = (spread - spread.rolling(self.lookback).mean()) / spread.rolling(self.lookback).std()
        return pd.DataFrame({
            'spread': spread,
            'zscore': zscore
        })

    def generate_signals(self, x: str, y: str, entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> pd.DataFrame:

        zdf = self.calculate_spread_and_zscore(x, y)
        zdf['signal'] = 0

        zdf.loc[zdf['zscore'] > entry_threshold, 'signal'] = -1  # short
        zdf.loc[zdf['zscore'] < -entry_threshold, 'signal'] = 1  # long 
        zdf.loc[(zdf['zscore'].abs() < exit_threshold), 'signal'] = 0 

        zdf['position'] = zdf['signal'].replace(to_replace=0, method='ffill').fillna(0)
        return zdf

    def get_price_data(self) -> pd.DataFrame:
        return self.prices

    def get_cointegrated_pairs(self) -> List[Tuple[str, str]]:

        return self.cointegrated_pairs
