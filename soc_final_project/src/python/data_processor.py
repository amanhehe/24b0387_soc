import pandas as pd
import os

def load_data(file_path):
    """
    Loads stock data from a CSV file, parses dates, and handles missing values.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Cleaned stock data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Parse date column if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Rename all columns to lowercase for consistency
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        # Convert date to string format for C++ compatibility
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    # Handle missing values (forward fill, then drop any remaining)
    df = df.fillna(method='ffill').dropna()
    return df

def get_training_data():
    """
    Loads and returns the training data for Reliance stock.
    """
    return load_data(os.path.join('data', 'reliance_train.csv'))

def get_testing_data():
    """
    Loads and returns the testing data for Reliance stock.
    """
    return load_data(os.path.join('data', 'reliance_test.csv'))

if __name__ == "__main__":
    train_df = get_training_data()
    test_df = get_testing_data()
    print("Training data sample:")
    print(train_df.head())
    print("\nTesting data sample:")
    print(test_df.head())