#!/usr/bin/env python3
"""
Testing script for the neural network model
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from model import TradingNeuralNetwork
from data_processor import load_data
from trading_analyzer import TradingAnalyzer
import strategies
import numpy as np

def test_model():
    """Test the neural network model"""
    try:
        # Load test data
        test_data = load_data('data/reliance_test.csv')

        # Compute indicators and signals for test data
        test_data_dict = test_data.to_dict('records')
        macd_signals = strategies.macd_strategy(test_data_dict)
        rsi_signals = strategies.rsi_strategy(test_data_dict)
        supertrend_signals = strategies.supertrend_strategy(test_data_dict)

        # Add signals as columns for feature extraction
        test_data['macd_signal'] = macd_signals
        test_data['rsi_signal'] = rsi_signals
        test_data['supertrend_signal'] = supertrend_signals

        # Add technical features
        test_data['ma5'] = test_data['close'].rolling(window=5).mean()
        test_data['ma10'] = test_data['close'].rolling(window=10).mean()
        test_data['ret1'] = test_data['close'].pct_change(1)
        test_data['ret5'] = test_data['close'].pct_change(5)
        test_data = test_data.fillna(0)
        # Prepare features for NN (same as in feature_engineering.py)
        features = []
        for i in range(len(test_data)):
            row = test_data.iloc[i]
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
            features.append(feature_vec)
        X_test = np.array(features)
        # Scale features using test set stats (for demo; ideally use train stats)
        mean = X_test.mean(axis=0)
        std = X_test.std(axis=0) + 1e-8
        X_test = (X_test - mean) / std

        # Load trained model
        model = TradingNeuralNetwork.load_model(input_size=X_test.shape[1], model_path='models/trading_nn.pt')

        # Make predictions (1=take trade, 0=skip)
        nn_preds = TradingNeuralNetwork.predict(model, X_test, threshold=0.3)

        # Generate NN-based trading signals (simulate position management)
        nn_signals = []
        in_position = False
        for i, pred in enumerate(nn_preds):
            if pred == 1 and not in_position:
                nn_signals.append('BUY')
                in_position = True
            elif pred == 0 and in_position:
                nn_signals.append('SELL')
                in_position = False
            else:
                nn_signals.append('HOLD')

        # Prepare strategy signals dict
        strategy_signals = {
            'MACD': macd_signals,
            'RSI': rsi_signals,
            'Supertrend': supertrend_signals,
            'NeuralNetwork': nn_signals
        }

        # Analyze performance
        analyzer = TradingAnalyzer(test_data, strategy_signals)
        results = analyzer.analyze_all_strategies()

        # Display results
        print("\n📋 Test Results:\n")
        for strategy, metrics in results.items():
            print(f"{strategy}:\nSuccess Rate: {metrics['success_rate']:.2f}%,\nPer-Trade Return: {metrics['per_trade_return']:.4f},\nTotal Trades: {metrics['total_trades']},\nTotal Return: {metrics.get('total_return', 0):.4f}\n")
        print("✅ Testing completed!")
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(test_model()) 