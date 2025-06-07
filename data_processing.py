import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_csv_data(path):
    return pd.read_csv(path)


def merge_data(news_df, stock_df):
    news_df['date'] = pd.to_datetime(news_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    return pd.merge(
        news_df,
        stock_df,
        how='left',
        left_on='date',
        right_on='date'
    )

