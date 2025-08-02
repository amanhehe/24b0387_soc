#!/usr/bin/env python3
"""
Neural Network Model for Trading Signal Improvement (PyTorch version)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TradingNeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    @staticmethod
    def train_model(X: np.ndarray, y: np.ndarray, input_size: int = 9, epochs: int = 200, lr: float = 0.001, save_path: str = 'models/trading_nn.pt', seed: int = 42):
        # Set random seed for reproducible training
        set_seed(seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TradingNeuralNetwork(input_size=input_size).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        return model

    @staticmethod
    def load_model(input_size: int = 9, model_path: str = 'models/trading_nn.pt'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TradingNeuralNetwork(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    @staticmethod
    def predict(model, X: np.ndarray, threshold: float = 0.3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor).cpu().numpy().flatten()
        preds = (outputs >= threshold).astype(int)
        return preds 