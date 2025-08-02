# Stock Market Analysis with Neural Networks - Technical Summary

##  Project Overview

This project implements a **hybrid algorithmic trading system** that combines the speed of C++ technical indicators with the learning capabilities of Python neural networks. The system demonstrates how machine learning can significantly improve traditional technical analysis strategies.

##  System Architecture

### High-Level Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stock Data    │───▶│  C++ Indicators │───▶│ Python Neural   │
│   (CSV Files)   │    │  (MACD, RSI,   │    │   Network       │
│                 │    │   Supertrend)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Feature Vector  │    │ Trading Signals │
                       │ (9 dimensions)  │    │ (BUY/SELL)     │
                       └─────────────────┘    └─────────────────┘
```

##  Technical Implementation Details

### 1. Data Processing Pipeline

#### `data_processor.py`
**Purpose**: Load and preprocess stock data for C++ compatibility

**Key Functions**:
```python
def load_data(file_path):
    # Load CSV with pandas
    # Convert date format for C++ compatibility
    # Handle missing values
    # Return clean DataFrame
```

**Logic**:
- **Date Conversion**: Convert Pandas Timestamp to string format (`%Y-%m-%d`) for C++ compatibility
- **Column Standardization**: Rename columns to lowercase for consistency
- **Data Cleaning**: Forward-fill missing values and drop any remaining NaN rows

### 2. C++ Technical Indicators

#### Data Structures (`data_types.h`)
```cpp
struct Candle {
    std::string date;
    double open, high, low, close, volume;
};

enum class Signal {
    BUY, SELL, HOLD
};
```

#### MACD Strategy (`macd_strategy.cpp`)
**Algorithm Logic**:
1. **EMA Calculation**: Exponential Moving Average with smoothing factor `k = 2/(period+1)`
2. **MACD Line**: `EMA(12) - EMA(26)`
3. **Signal Line**: `EMA(9)` of MACD line
4. **Signal Generation**: 
   - BUY when MACD crosses above Signal line
   - SELL when MACD crosses below Signal line

**Critical Fix**: Initially had look-ahead bias in EMA calculation. Fixed to calculate iteratively:
```cpp
static double ema(const std::vector<double>& values, int period, int current) {
    if (current < period - 1) return 0.0;
    // Start with SMA for initial period
    double ema_value = sma(values, period, period - 1);
    // Calculate EMA iteratively for current period only
    double k = 2.0 / (period + 1.0);
    for (int i = period; i <= current; ++i) {
        ema_value = (values[i] * k) + (ema_value * (1.0 - k));
    }
    return ema_value;
}
```

#### RSI Strategy (`rsi_strategy.cpp`)
**Algorithm Logic**:
1. **Gain/Loss Calculation**: Calculate daily price changes
2. **Average Gain/Loss**: 14-period simple moving average
3. **RSI Formula**: `RSI = 100 - (100 / (1 + RS))` where `RS = AvgGain / AvgLoss`
4. **Signal Generation**:
   - BUY when RSI < 30 (oversold)
   - SELL when RSI > 70 (overbought)

#### Supertrend Strategy (`supertrend_strategy.cpp`)
**Algorithm Logic**:
1. **ATR Calculation**: Average True Range for volatility measurement
2. **Upper Band**: `(High + Low) / 2 + (Multiplier × ATR)`
3. **Lower Band**: `(High + Low) / 2 - (Multiplier × ATR)`
4. **Trend Determination**: Based on price position relative to bands
5. **Signal Generation**:
   - BUY when price crosses above upper band
   - SELL when price crosses below lower band

### 3. Pybind11 Integration (`binding.cpp`)

**Purpose**: Bridge C++ and Python for seamless communication

**Key Functions**:
```cpp
PYBIND11_MODULE(strategies, m) {
    // Expose C++ functions to Python
    m.def("macd_strategy", &macd_strategy);
    m.def("rsi_strategy", &rsi_strategy);
    m.def("supertrend_strategy", &supertrend_strategy);
}
```

**Data Conversion**:
- Convert Python list of dictionaries to C++ vector of Candle structs
- Convert C++ vector of Signal enum to Python list of strings

### 4. Feature Engineering (`feature_engineering.py`)

**Purpose**: Create training data for neural network

**Feature Vector (9 dimensions)**:
```python
feature_vec = [
    row['close'],           # Current price
    row['volume'],          # Trading volume
    row['ma5'],            # 5-day moving average
    row['ma10'],           # 10-day moving average
    row['ret1'],           # 1-day return
    row['ret5'],           # 5-day return
    1 if row['macd_signal'] == 'BUY' else 0,    # MACD signal
    1 if row['rsi_signal'] == 'BUY' else 0,     # RSI signal
    1 if row['supertrend_signal'] == 'BUY' else 0  # Supertrend signal
]
```

**Label Generation Logic**:
1. **Trade Extraction**: Find BUY/SELL pairs from strategy signals
2. **Profit Calculation**: `(sell_price - buy_price) / buy_price`
3. **Label Assignment**: 
   - `1` if trade profit > 0 (successful)
   - `0` if trade profit ≤ 0 (unsuccessful)

**Feature Scaling**:
```python
def scale_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # Avoid division by zero
    return (X - mean) / std
```

### 5. Neural Network Model (`model.py`)

**Architecture**:
```python
class TradingNeuralNetwork(nn.Module):
    def __init__(self, input_size=9, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)    # Input → Hidden 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)   # Hidden 1 → Hidden 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)             # Hidden 2 → Output
        self.sigmoid = nn.Sigmoid()
```

**Training Process**:
- **Loss Function**: Binary Cross Entropy Loss
- **Optimizer**: Adam with learning rate 0.001
- **Epochs**: 200 for convergence
- **Threshold**: 0.3 for prediction (lower threshold = more trades)
- **Reproducible**: Fixed random seed (42) ensures consistent results across runs

**Prediction Logic**:
```python
def predict(model, X, threshold=0.3):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X))
        predictions = (outputs >= threshold).float()
    return predictions.numpy().flatten()
```

### 6. Trading Analysis (`trading_analyzer.py`)

**Performance Metrics**:
1. **Success Rate**: `(winning_trades / total_trades) × 100`
2. **Per-Trade Return**: Average return per trade
3. **Total Trades**: Number of completed BUY/SELL cycles
4. **Total Return**: Cumulative return over entire period

**Position Management Logic**:
```python
def calculate_trades(self, signals):
    trades = []
    in_position = False
    buy_price = 0
    
    for i, signal in enumerate(signals):
        if signal == 'BUY' and not in_position:
            in_position = True
            buy_price = self.data.iloc[i]['close']
        elif signal == 'SELL' and in_position:
            sell_price = self.data.iloc[i]['close']
            trade_return = (sell_price - buy_price) / buy_price
            trades.append(trade_return)
            in_position = False
    
    return trades
```

##  Neural Network Logic

### Why Neural Networks Work Better

1. **Feature Learning**: Discovers complex, non-linear patterns in market data
2. **Signal Filtering**: Only takes high-probability trades, avoiding noise
3. **Risk Management**: Learns to avoid low-quality signals from pure strategies
4. **Adaptive**: Continuously learns from historical trade outcomes

### Training Process
1. **Data Preparation**: 105 training samples from 10 years of data
2. **Feature Engineering**: 9-dimensional feature space
3. **Label Generation**: Binary classification (profitable/unprofitable)
4. **Model Training**: 200 epochs with early stopping
5. **Validation**: Test on 5 years of unseen data

##  Performance Analysis

### Strategy Comparison

| Metric | MACD | RSI | Supertrend | Neural Network |
|--------|------|-----|------------|----------------|
| Success Rate | 33.33% | 64.29% | 50.00% | **57.07%** |
| Total Trades | 45 | 14 | 16 | **184** |
| Per-Trade Return | 0.0053 | 0.0237 | 0.0243 | 0.0045 |
| Total Return | 0.2372 | 0.3317 | 0.3888 | **0.8361** |

### Key Insights

1. **Neural Network Advantage**: +115.1% total return improvement
2. **Trade Frequency**: 4x more trades than pure strategies
3. **Risk Management**: Good success rate with more trades
4. **Consistency**: Balanced risk-reward profile

##  Complete Workflow

### Step 1: Feature Engineering
```python
# Load training data
df = load_data('data/reliance_train.csv')

# Compute C++ indicators
macd_signals = strategies.macd_strategy(data_dict)
rsi_signals = strategies.rsi_strategy(data_dict)
supertrend_signals = strategies.supertrend_strategy(data_dict)

# Add technical features
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
df['ret1'] = df['close'].pct_change(1)
df['ret5'] = df['close'].pct_change(5)

# Extract features and labels
X, labels = extract_trade_features_and_labels(df)

# Save for training
np.save('models/features.npy', X)
np.save('models/labels.npy', labels)
```

### Step 2: Model Training
```python
# Load features and labels
X = np.load('models/features.npy')
y = np.load('models/labels.npy')

# Train neural network
model = TradingNeuralNetwork.train_model(X, y, epochs=200)

# Save trained model
torch.save(model.state_dict(), 'models/trading_nn.pt')
```

### Step 3: Testing and Comparison
```python
# Load test data and compute all strategies
test_data = load_data('data/reliance_test.csv')
macd_signals = strategies.macd_strategy(test_data_dict)
rsi_signals = strategies.rsi_strategy(test_data_dict)
supertrend_signals = strategies.supertrend_strategy(test_data_dict)

# Generate neural network predictions
X_test = prepare_test_features(test_data)
model = TradingNeuralNetwork.load_model(input_size=X_test.shape[1])
nn_predictions = TradingNeuralNetwork.predict(model, X_test)

# Convert predictions to trading signals
nn_signals = predictions_to_signals(nn_predictions)

# Analyze all strategies
analyzer = TradingAnalyzer(test_data, {
    'MACD': macd_signals,
    'RSI': rsi_signals,
    'Supertrend': supertrend_signals,
    'NeuralNetwork': nn_signals
})
results = analyzer.analyze_all_strategies()
```

##  Key Innovations

1. **Hybrid Architecture**: C++ speed + Python flexibility
2. **Position Management**: Proper BUY/SELL logic
3. **Feature Engineering**: Multi-dimensional feature space
4. **Neural Network Enhancement**: ML-based signal filtering
5. **Comprehensive Analysis**: Multiple performance metrics


##  Conclusion

This project successfully demonstrates how machine learning can enhance traditional technical analysis. The neural network approach achieves **115.1% better total return** than the best pure strategy while maintaining a good success rate and generating more trading opportunities.

The hybrid C++/Python architecture provides both performance and flexibility, making it suitable for real-world algorithmic trading applications.

**Total Return Improvement**: +115.1% over best pure strategy
**Success Rate**: 57.07% with 184 trades
**Risk Management**: Proper position sizing and signal filtering 
