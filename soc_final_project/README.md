# Stock Market Analysis with Neural Networks

A comprehensive algorithmic trading system that combines fast C++ technical indicators with Python neural networks to improve trading signals. This project demonstrates the power of hybrid approaches in algorithmic trading.

##  Project Overview

This system uses **three technical indicators** (MACD, RSI, Supertrend) to generate trading signals, then employs a **Neural Network** to filter and improve these signals, resulting in significantly better performance than pure technical strategies.

##  Performance Results

| Strategy | Success Rate | Per-Trade Return | Total Trades | Total Return |
|----------|-------------|------------------|--------------|--------------|
| MACD | 33.33% | 0.0053 | 45 | 0.2372 |
| RSI | 64.29% | 0.0237 | 14 | 0.3317 |
| Supertrend | 50.00% | 0.0243 | 16 | 0.3888 |
| **Neural Network** | **57.07%** | **0.0045** | **184** | **0.8361** |

**Neural Network Improvement**: +115.1% total return over best pure strategy!

##  Prerequisites

- **Python 3.8 or higher**
- **C++ compiler** (Visual Studio on Windows, GCC on Linux)
- **Git** (optional, for cloning)

##  Installation & Setup

### Step 1: Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv trading_env

# Activate the virtual environment
# On Windows:
trading_env\Scripts\activate

# On Linux/Mac:
source trading_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Build C++ Extensions

```bash
# Build the C++ strategies module
python setup.py build_ext --inplace
```

## 📁 Project Structure

```
stockMarketAnalysis-usingNN/
├── 📁 data/                    # Stock data files
│   ├── reliance_train.csv      # Training data (10 years)
│   └── reliance_test.csv       # Test data (5 years)
├── 📁 src/                     # Source code
│   ├── 📁 cpp/                 # C++ trading strategies
│   │   ├── binding.cpp         # Pybind11 bindings
│   │   ├── data_types.h        # Common data structures
│   │   ├── macd_strategy.h/cpp # MACD implementation
│   │   ├── rsi_strategy.h/cpp  # RSI implementation
│   │   └── supertrend_strategy.h/cpp # Supertrend implementation
│   └── 📁 python/             # Python modules
│       ├── data_processor.py   # Data loading and preprocessing
│       ├── feature_engineering.py # Feature extraction for NN
│       ├── model.py            # Neural network model (PyTorch)
│       ├── train.py            # Training script
│       ├── test.py             # Testing and comparison script
│       └── trading_analyzer.py # Performance analysis
├── 📁 models/                  # Generated models (created during execution)
├── 📁 results/                 # Performance results (created during execution)
├── 📄 README.md                # This file
├── 📄 PROJECT_SUMMARY.md       # Detailed technical explanation
├── 📄 requirements.txt         # Python dependencies
├── 📄 config.py                # Project configuration
└── 📄 setup.py                 # Pybind11 build configuration
```

##  How to Run the Project

### Complete Pipeline (Recommended)

```bash
# 1. Generate features from training data
python src/python/feature_engineering.py

# 2. Train the neural network
python src/python/train.py

# 3. Test and compare all strategies
python src/python/test.py
```

### Individual Steps

#### Step 1: Feature Engineering
```bash
python src/python/feature_engineering.py
```
**What it does:**
- Loads training data from `data/reliance_train.csv`
- Computes C++ technical indicators (MACD, RSI, Supertrend)
- Extracts features for neural network training
- Saves features and labels to `models/` directory

#### Step 2: Train Neural Network
```bash
python src/python/train.py
```
**What it does:**
- Loads features and labels from `models/` directory
- Trains a PyTorch neural network (2 hidden layers, 64 units each)
- Saves trained model to `models/trading_nn.pt`

#### Step 3: Test and Compare Strategies
```bash
python src/python/test.py
```
**What it does:**
- Loads test data from `data/reliance_test.csv`
- Computes signals for all strategies (MACD, RSI, Supertrend, Neural Network)
- Analyzes performance metrics
- Displays comparison results

##  Technical Details

### C++ Strategies
- **MACD**: Moving Average Convergence Divergence with EMA12-EMA26 crossover
- **RSI**: Relative Strength Index with overbought/oversold signals
- **Supertrend**: Trend-following indicator with band calculations

### Neural Network Architecture
- **Input Layer**: 9 features (price, volume, indicators, signals)
- **Hidden Layer 1**: 64 units with ReLU activation
- **Hidden Layer 2**: 64 units with ReLU activation
- **Output Layer**: 1 unit with Sigmoid activation
- **Training**: 200 epochs with Adam optimizer
- **Reproducible**: Fixed random seed (42) for consistent results

### Feature Engineering
```
Input Features (9 dimensions):
├── Price data (close, volume)
├── Technical indicators (MA5, MA10, returns)
└── Strategy signals (MACD, RSI, Supertrend BUY signals)
```

##  Expected Output

After running the complete pipeline, you'll see:

```
 Test Results:

MACD:
Success Rate: 33.33%,
Per-Trade Return: 0.0053,
Total Trades: 45,
Total Return: 0.2372

RSI:
Success Rate: 64.29%,
Per-Trade Return: 0.0237,
Total Trades: 14,
Total Return: 0.3317

Supertrend:
Success Rate: 50.00%,
Per-Trade Return: 0.0243,
Total Trades: 16,
Total Return: 0.3888

NeuralNetwork:
Success Rate: 57.07%,
Per-Trade Return: 0.0045,
Total Trades: 184,
Total Return: 0.8361

 Testing completed!
```

##  Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **C++ compilation errors**
   - Install Visual Studio Build Tools (Windows)
   - Install GCC (Linux/Mac)

3. **"No module named 'strategies'"**
   - Run `python setup.py build_ext --inplace` first


### File Dependencies

- **Training Data**: `data/reliance_train.csv` (required)
- **Test Data**: `data/reliance_test.csv` (required)
- **Generated Files**: Created automatically during execution

##  Learning Objectives

This project demonstrates:
- **C++/Python Integration** using pybind11
- **Technical Analysis** with multiple indicators
- **Machine Learning** for signal improvement
- **Backtesting** and performance analysis
- **Software Engineering** best practices

