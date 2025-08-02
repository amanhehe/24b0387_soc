import pandas as pd
import numpy as np
import sys
import os
import random
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_processor import load_data
import strategies
import config

# Set random seed for reproducible results
def set_seed(seed=42):
    """Set random seeds for reproducible results"""
    np.random.seed(seed)
    random.seed(seed)

def compute_indicators(df):
    """
    Compute MACD, RSI, Supertrend using C++ strategies and add as columns.
    """
    data_dict = df.to_dict('records')
    df['macd_signal'] = strategies.macd_strategy(data_dict)
    df['rsi_signal'] = strategies.rsi_strategy(data_dict)
    df['supertrend_signal'] = strategies.supertrend_strategy(data_dict)
    return df

def add_technical_features(df):
    """
    Add moving averages, returns, and scale features.
    """
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ret1'] = df['close'].pct_change(1)
    df['ret5'] = df['close'].pct_change(5)
    # Fill NaNs with 0 for early rows
    df = df.fillna(0)
    return df

def scale_features(X):
    """
    Standardize features (mean=0, std=1)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std

def extract_trade_features_and_labels(df, signals_col='macd_signal'):
    features = []
    labels = []
    in_position = False
    buy_index = None
    signals = df[signals_col].values
    for i, signal in enumerate(signals):
        if signal == 'BUY' and not in_position:
            in_position = True
            buy_index = i
        elif signal == 'SELL' and in_position:
            row = df.iloc[buy_index]
            feature_vec = [
                row['close'],
                row['volume'],
                row['ma5'],
                row['ma10'],
                row['ret1'],
                row['ret5'],
                1 if row['macd_signal'] == 'BUY' else 0,
                1 if row['rsi_signal'] == 'BUY' else 0,
                1 if row['supertrend_signal'] == 'BUY' else 0,
            ]
            buy_price = row['close']
            sell_price = df.iloc[i]['close']
            label = 1 if sell_price > buy_price else 0
            features.append(feature_vec)
            labels.append(label)
            in_position = False
            buy_index = None
    X = np.array(features)
    X = scale_features(X)
    return X, np.array(labels)

if __name__ == "__main__":
    # Set random seed for reproducible results
    set_seed(config.RANDOM_SEED)
    
    df = load_data('data/reliance_train.csv')
    df = compute_indicators(df)
    df = add_technical_features(df)
    X, labels = extract_trade_features_and_labels(df, signals_col='macd_signal')
    print(f"Generated {len(X)} samples for NN training.")
    np.save('models/features.npy', X)
    np.save('models/labels.npy', labels)
    print("Saved features and labels to models/ directory.")