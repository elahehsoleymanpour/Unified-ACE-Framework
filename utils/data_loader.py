import pandas as pd
import numpy as np

def load_and_preprocess_unified(filepath):
    """Aggregates news headlines by date and aligns them with market returns."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df_grouped = df.groupby('Date').agg({'Title': list, 'CP': 'last'}).reset_index()
    df_grouped = df_grouped.sort_values('Date')
    
    prices = df_grouped['CP'].values
    dates = df_grouped['Date'].values[1:]
    news_lists = df_grouped['Title'].values[1:]
    returns = np.log(prices[1:] / prices[:-1])
    return returns, dates, news_lists
