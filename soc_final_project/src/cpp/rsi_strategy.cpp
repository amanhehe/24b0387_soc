#include "rsi_strategy.h"

std::vector<Signal> rsi_signals(const std::vector<Candle>& candles, int period, double overbought, double oversold) {
    std::vector<Signal> signals(candles.size(), Signal::HOLD);
    std::vector<double> gains, losses;
    for (size_t i = 1; i < candles.size(); ++i) {
        double change = candles[i].close - candles[i-1].close;
        gains.push_back(change > 0 ? change : 0);
        losses.push_back(change < 0 ? -change : 0);
    }
    
    bool in_position = false; // Track if we're currently holding a position
    
    for (size_t i = period; i < gains.size(); ++i) {
        double avg_gain = 0, avg_loss = 0;
        for (size_t j = i - period; j < i; ++j) {
            avg_gain += gains[j];
            avg_loss += losses[j];
        }
        avg_gain /= period;
        avg_loss /= period;
        double rs = avg_loss == 0 ? 100 : avg_gain / avg_loss;
        double rsi = 100 - (100 / (1 + rs));
        
        if (!in_position && rsi < oversold) {
            signals[i+1] = Signal::BUY;
            in_position = true;
        }
        else if (in_position && rsi > overbought) {
            signals[i+1] = Signal::SELL;
            in_position = false;
        }
        // If no crossover, keep current position (HOLD)
    }
    return signals;
}