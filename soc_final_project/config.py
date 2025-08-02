#!/usr/bin/env python3
"""
Configuration file for Stock Market Analysis project
"""

# Data paths
TRAIN_DATA_PATH = "data/reliance_train.csv"
TEST_DATA_PATH = "data/reliance_test.csv"

# Model paths
MODEL_PATH = "models/trading_nn.pt"
FEATURES_PATH = "models/features.npy"
LABELS_PATH = "models/labels.npy"

# Results path
RESULTS_PATH = "results/strategy_performance.csv"

# Neural Network parameters
NN_INPUT_SIZE = 9
NN_HIDDEN_SIZE = 64
NN_EPOCHS = 200
NN_LEARNING_RATE = 0.001
NN_THRESHOLD = 0.3
RANDOM_SEED = 42  # For reproducible results

# Feature engineering parameters
MA_PERIODS = [5, 10]  # Moving average periods
RETURN_PERIODS = [1, 5]  # Return calculation periods

# Trading parameters
POSITION_MANAGEMENT = True  # Enable position management logic 