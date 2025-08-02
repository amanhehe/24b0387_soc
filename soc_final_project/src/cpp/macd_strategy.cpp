#include "macd_strategy.h"

// Helper to calculate SMA (Simple Moving Average)
static double sma(const std::vector<double>& values, int period, int current) {
    if (current < period - 1) return 0.0;
    double sum = 0.0;
    for (int i = current - period + 1; i <= current; ++i) {
        sum += values[i];
    }
    return sum / period;
}

// Helper to calculate EMA (Exponential Moving Average)
static double ema(const std::vector<double>& values, int period, int current) {
    if (current < period - 1) return 0.0;
    
    // Start with SMA for the first period
    double ema_value = sma(values, period, period - 1);
    
    // Calculate EMA for the current period only
    double k = 2.0 / (period + 1.0);
    for (int i = period; i <= current; ++i) {
        ema_value = (values[i] * k) + (ema_value * (1.0 - k));
    }
    
    return ema_value;
}

// Helper to calculate EMA of MACD line (for signal line)
static double ema_macd(const std::vector<double>& macd_values, int period, int current) {
    if (current < period - 1) return 0.0;
    
    // Start with SMA of MACD for the first period
    double ema_value = sma(macd_values, period, period - 1);
    
    // Calculate EMA for the current period only
    double k = 2.0 / (period + 1.0);
    for (int i = period; i <= current; ++i) {
        ema_value = (macd_values[i] * k) + (ema_value * (1.0 - k));
    }
    
    return ema_value;
}

std::vector<Signal> macd_signals(const std::vector<Candle>& candles) {
    std::vector<Signal> signals(candles.size(), Signal::HOLD);
    std::vector<double> closes;
    for (const auto& c : candles) closes.push_back(c.close);

    // Calculate EMAs for all periods
    std::vector<double> ema12_values, ema26_values;
    for (size_t i = 0; i < closes.size(); ++i) {
        ema12_values.push_back(ema(closes, 12, i));
        ema26_values.push_back(ema(closes, 26, i));
    }
    
    // Calculate MACD line
    std::vector<double> macd_line;
    for (size_t i = 0; i < closes.size(); ++i) {
        if (ema12_values[i] > 0 && ema26_values[i] > 0) {
            macd_line.push_back(ema12_values[i] - ema26_values[i]);
        } else {
            macd_line.push_back(0.0);
        }
    }
    
    // Calculate signal line (EMA9 of MACD)
    std::vector<double> signal_line;
    for (size_t i = 0; i < macd_line.size(); ++i) {
        signal_line.push_back(ema_macd(macd_line, 9, i));
    }

    bool in_position = false; // Track if we're currently holding a position
    
    for (size_t i = 26; i < candles.size(); ++i) {
        if (macd_line[i] != 0.0 && signal_line[i] != 0.0) {
            if (!in_position && macd_line[i] > signal_line[i]) {
                signals[i] = Signal::BUY;
                in_position = true;
            }
            else if (in_position && macd_line[i] < signal_line[i]) {
                signals[i] = Signal::SELL;
                in_position = false;
            }
        }
    }
    return signals;
}