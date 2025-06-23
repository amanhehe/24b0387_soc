
#include "fibonacci_strategy.h"
#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

TradeResult run_fibonacci_strategy(const vector<Candle> &candles, double profit_threshold, int lookback) {
    if (candles.size() < static_cast<size_t>(lookback) + 1) {
        cerr << "Insufficient data: Need at least " << lookback + 1 << " candles" << endl;
        return {0.0, 0.0, 0, {}};
    }

    vector<double> closes;
    for (const auto &candle : candles)
        closes.push_back(candle.close);

    vector<int> fib_positions(closes.size(), 0);
    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;
    double entry_price = 0.0;
    enum Position { NONE, LONG, SHORT } state = NONE;

    for (size_t i = lookback; i < closes.size(); ++i) {
        // Proper high/low calculation
        double recent_low = numeric_limits<double>::max();
        double recent_high = numeric_limits<double>::min();
        
        for (size_t j = i - lookback; j < i; ++j) {
            if (closes[j] < recent_low) recent_low = closes[j];
            if (closes[j] > recent_high) recent_high = closes[j];
        }

        double diff = recent_high - recent_low;
        vector<double> levels = {
            recent_high - 0.236 * diff,  // 23.6%
            recent_high - 0.382 * diff,  // 38.2%
            recent_high - 0.500 * diff,  // 50.0%
            recent_high - 0.618 * diff,  // 61.8%
            recent_high - 0.786 * diff   // 78.6%
        };

        double price = closes[i];
        double prev_price = closes[i-1];

        // Entry signals with trend confirmation
        if (state == NONE) {
            // Uptrend retracement (long entry)
            if (price > levels[3] && prev_price <= levels[3] && price > levels[0]) {
                state = LONG;
                entry_price = price;
                fib_positions[i] = 1;
            }
            // Downtrend retracement (short entry)
            else if (price < levels[3] && prev_price >= levels[3] && price < levels[0]) {
                state = SHORT;
                entry_price = price;
                fib_positions[i] = -2;
            }
        }
        // Exit conditions
        else if (state == LONG) {
            double ret = (price - entry_price) / entry_price;
            // Take profit at resistance or stop loss
            if (ret > profit_threshold || price < levels[4] || price >= levels[0]) {
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                fib_positions[i] = -1;
                state = NONE;
            }
        }
        else if (state == SHORT) {
            double ret = (entry_price - price) / entry_price;
            // Take profit at support or stop loss
            if (ret > profit_threshold || price > levels[4] || price <= levels[0]) {
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                fib_positions[i] = 2;
                state = NONE;
            }
        }
    }

    // Close open positions
    if (state != NONE) {
        double final_price = closes.back();
        double ret = (state == LONG) 
                   ? (final_price - entry_price) / entry_price
                   : (entry_price - final_price) / entry_price;
        total_return += ret;
        profitable_trades += (ret > profit_threshold) ? 1 : 0;
        total_trades++;
        fib_positions.back() = (state == LONG) ? -1 : 2;
    }

    double success_rate = total_trades > 0
                        ? (static_cast<double>(profitable_trades) / total_trades) * 100
                        : 0.0;
    double avg_return = total_trades > 0
                      ? (total_return / total_trades) * 100
                      : 0.0;

    return {success_rate, avg_return, total_trades, fib_positions};
}
