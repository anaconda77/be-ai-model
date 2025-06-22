import pandas as pd


def get_csv_data(path):
    return pd.read_csv(path)


def merge_data(news_df, stock_df):
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.normalize()

    return pd.merge(
        news_df,
        stock_df,
        how="left",
        on="date",
    )
