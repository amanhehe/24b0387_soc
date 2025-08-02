#include "supertrend_strategy.h"
#include <vector>
#include <algorithm>

std::vector<Signal> supertrend_signals(const std::vector<Candle>& candles, int period, double multiplier) {
    std::vector<Signal> signals(candles.size(), Signal::HOLD);
    if (candles.size() < static_cast<size_t>(period)) return signals;
    std::vector<double> atr(candles.size(), 0.0);
    // Calculate ATR
    for (size_t i = 1; i < candles.size(); ++i) {
        double tr = std::max({candles[i].high - candles[i].low,
                              std::abs(candles[i].high - candles[i-1].close),
                              std::abs(candles[i].low - candles[i-1].close)});
        if (i < static_cast<size_t>(period)) {
            atr[i] = atr[i-1] + tr;
        } else {
            atr[i] = (atr[i-1] * (static_cast<size_t>(period) - 1) + tr) / period;
        }
    }
    std::vector<double> upperband(candles.size(), 0.0), lowerband(candles.size(), 0.0);
    std::vector<bool> in_uptrend(candles.size(), true);
    bool in_position = false; // Track if we're currently holding a position
    
    for (size_t i = static_cast<size_t>(period); i < candles.size(); ++i) {
        double hl2 = (candles[i].high + candles[i].low) / 2.0;
        double basic_upper = hl2 + multiplier * atr[i];
        double basic_lower = hl2 - multiplier * atr[i];
        
        // Proper Supertrend logic with band continuity
        if (i == static_cast<size_t>(period)) {
            upperband[i] = basic_upper;
            lowerband[i] = basic_lower;
        } else {
            // Final upper band - only change if basic_upper is lower than previous OR if close crossed above previous upper
            if (basic_upper < upperband[i-1] || candles[i-1].close > upperband[i-1]) {
                upperband[i] = basic_upper;
            } else {
                upperband[i] = upperband[i-1];
            }
            
            // Final lower band - only change if basic_lower is higher than previous OR if close crossed below previous lower
            if (basic_lower > lowerband[i-1] || candles[i-1].close < lowerband[i-1]) {
                lowerband[i] = basic_lower;
            } else {
                lowerband[i] = lowerband[i-1];
            }
        }
        
        // Determine trend based on previous bands (standard Supertrend logic)
        if (candles[i].close > upperband[i-1]) {
            in_uptrend[i] = true;
        } else if (candles[i].close < lowerband[i-1]) {
            in_uptrend[i] = false;
        } else {
            in_uptrend[i] = in_uptrend[i-1];
        }
        
        if (!in_position && in_uptrend[i]) {
            signals[i] = Signal::BUY;
            in_position = true;
        }
        else if (in_position && !in_uptrend[i]) {
            signals[i] = Signal::SELL;
            in_position = false;
        }
        // If no crossover, keep current position (HOLD)
    }
    return signals;
}