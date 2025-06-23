#include "macd_strategy.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

vector<double> calculate_ema_vector(const vector<double> &values, int period) {
    if (values.size() < period) 
        return vector<double>(values.size(), NAN);

    vector<double> ema(values.size(), 0.0);
    double k = 2.0 / (period + 1.0);

    double sum = 0.0;
    for (int i = 0; i < period; ++i) {
        sum += values[i];
    }
    ema[period - 1] = sum / period;

    for (int i = period; i < values.size(); ++i) {
        ema[i] = values[i] * k + ema[i - 1] * (1 - k);
    }
    return ema;
}

TradeResult run_macd_strategy(const vector<Candle> &candles, double profit_threshold) {
    if (candles.size() < 35) {
        cerr << "Insufficient data: Need at least 35 candles" << endl;
        return {0.0, 0.0, 0, {}};
    }

    vector<double> closes;
    for (const auto &candle : candles)
        closes.push_back(candle.close);

    vector<int> macd_positions(closes.size(), 0);

    vector<double> ema12 = calculate_ema_vector(closes, 12);
    vector<double> ema26 = calculate_ema_vector(closes, 26);

    vector<double> macd_line(closes.size(), 0.0);
    for (size_t i = 0; i < closes.size(); ++i) {
        if (i >= 25)
            macd_line[i] = ema12[i] - ema26[i];
    }

    vector<double> signal_line = calculate_ema_vector(macd_line, 9);

    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;
    double entry_price = 0.0;
    enum Position { NONE, LONG, SHORT } state = NONE;

    for (size_t i = 34; i < closes.size(); ++i) {
        double prev_macd = macd_line[i - 1];
        double prev_signal = signal_line[i - 1];
        double curr_macd = macd_line[i];
        double curr_signal = signal_line[i];

        if (state == NONE) {
            if (prev_macd < prev_signal && curr_macd > curr_signal) {
                state = LONG;
                entry_price = closes[i];
                macd_positions[i] = 1;
            }
            else if (prev_macd > prev_signal && curr_macd < curr_signal) {
                state = SHORT;
                entry_price = closes[i];
                macd_positions[i] = -2;
            }
        }
        else if (state == LONG) {
            if (prev_macd > prev_signal && curr_macd < curr_signal) {
                double exit_price = closes[i];
                double ret = (exit_price - entry_price) / entry_price;
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                state = NONE;
                macd_positions[i] = -1;
            }
        }
        else if (state == SHORT) {
            if (prev_macd < prev_signal && curr_macd > curr_signal) {
                double exit_price = closes[i];
                double ret = (entry_price - exit_price) / entry_price;
                total_return += ret;
                profitable_trades += (ret > profit_threshold) ? 1 : 0;
                total_trades++;
                state = NONE;
                macd_positions[i] = 2;
            }
        }
    }

    if (state != NONE) {
        double final_price = closes.back();
        double ret = (state == LONG) 
                   ? (final_price - entry_price) / entry_price
                   : (entry_price - final_price) / entry_price;
        total_return += ret;
        profitable_trades += (ret > profit_threshold) ? 1 : 0;
        total_trades++;
        macd_positions.back() = (state == LONG) ? -1 : 2;
    }

    double success_rate = total_trades > 0 
                        ? (static_cast<double>(profitable_trades) / total_trades) * 100 
                        : 0.0;
    double avg_return = total_trades > 0 
                      ? (total_return / total_trades) * 100 
                      : 0.0;

    return {success_rate, avg_return, total_trades, macd_positions};
}
