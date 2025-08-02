#ifndef RSI_STRATEGY_H
#define RSI_STRATEGY_H

#include <vector>
#include "data_types.h"

std::vector<Signal> rsi_signals(const std::vector<Candle>& candles, int period = 14, double overbought = 70.0, double oversold = 30.0);

#endif // RSI_STRATEGY_H