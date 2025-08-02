#!/usr/bin/env python3
"""
Training script for the neural network model
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from model import TradingNeuralNetwork
import numpy as np
import config

def train_model():
    try:
        X = np.load('models/features.npy')
        y = np.load('models/labels.npy')
        model = TradingNeuralNetwork.train_model(
            X, y, 
            input_size=X.shape[1], 
            epochs=config.NN_EPOCHS, 
            lr=config.NN_LEARNING_RATE, 
            save_path=config.MODEL_PATH,
            seed=config.RANDOM_SEED
        )
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(train_model()) 