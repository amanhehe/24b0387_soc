#include "bollinger_strategy.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

vector<double> calculate_sma(const vector<double> &values, int period) {
    vector<double> sma(values.size(), NAN);
    if (values.size() < period) 
        return sma;

    double sum = 0.0;
    for (int i = 0; i < period; ++i) {
        sum += values[i];
    }
    sma[period-1] = sum / period;

    for (int i = period; i < values.size(); ++i) {
        sum += values[i] - values[i - period];
        sma[i] = sum / period;
    }
    return sma;
}

vector<double> calculate_std_dev(const vector<double> &values, int period) {
    vector<double> stddev(values.size(), NAN);
    if (values.size() < period) 
        return stddev;

    vector<double> sma = calculate_sma(values, period);

    for (int i = period - 1; i < values.size(); ++i) {
        double variance = 0.0;
        for (int j = i - period + 1; j <= i; ++j) {
            variance += pow(values[j] - sma[j], 2); // Use SMA at time j
        }
        stddev[i] = sqrt(variance / period);
    }
    return stddev;
}

TradeResult run_bollinger_strategy(const vector<Candle> &candles, double profit_threshold) {
    const int period = 20;
    const double num_std_dev = 2.0;

    if (candles.size() < period) {
        cerr << "Insufficient data: Need at least " << period << " candles" << endl;
        return {0.0, 0.0, 0, {}};
    }

    vector<double> closes;
    for (const auto &candle : candles)
        closes.push_back(candle.close);

    vector<int> bollinger_positions(closes.size(), 0);
    vector<double> sma = calculate_sma(closes, period);
    vector<double> stddev = calculate_std_dev(closes, period); // Fixed calculation

    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;
    double entry_price = 0.0;
    enum Position { NONE, LONG, SHORT } state = NONE;

    for (size_t i = period; i < closes.size(); ++i) {
        double upper = sma[i] + num_std_dev * stddev[i];
        double lower = sma[i] - num_std_dev * stddev[i];

        if (state == NONE) {
            // Long entry: Price touches lower band
            if (closes[i] <= lower) {
                state = LONG;
                entry_price = closes[i];
                bollinger_positions[i] = 1;
            }
            // Short entry: Price touches upper band
            else if (closes[i] >= upper) {
                state = SHORT;
                entry_price = closes[i];
                bollinger_positions[i] = -2;
            }
        }
        else if (state == LONG) {
            // Exit long when price touches upper band OR crosses SMA upward
            if (closes[i] >= upper || closes[i] > sma[i]) {
                double exit_price = closes[i];
                double ret = (exit_price - entry_price) / entry_price;
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                state = NONE;
                bollinger_positions[i] = -1;
            }
        }
        else if (state == SHORT) {
            // Exit short when price touches lower band OR crosses SMA downward
            if (closes[i] <= lower || closes[i] < sma[i]) {
                double exit_price = closes[i];
                double ret = (entry_price - exit_price) / entry_price;
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                state = NONE;
                bollinger_positions[i] = 2;
            }
        }
    }

    // Close open positions at end
    if (state != NONE) {
        double final_price = closes.back();
        double ret = (state == LONG) 
                   ? (final_price - entry_price) / entry_price
                   : (entry_price - final_price) / entry_price;
        total_return += ret;
        profitable_trades += (ret > profit_threshold) ? 1 : 0;
        total_trades++;
        bollinger_positions.back() = (state == LONG) ? -1 : 2;
    }

    double success_rate = total_trades > 0 
                        ? (static_cast<double>(profitable_trades) / total_trades) * 100 
                        : 0.0;
    double avg_return = total_trades > 0 
                      ? (total_return / total_trades) * 100 
                      : 0.0;

    return {success_rate, avg_return, total_trades, bollinger_positions};
}
