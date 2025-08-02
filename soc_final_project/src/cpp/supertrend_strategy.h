#ifndef SUPERTREND_STRATEGY_H
#define SUPERTREND_STRATEGY_H

#include <vector>
#include "data_types.h"

std::vector<Signal> supertrend_signals(const std::vector<Candle>& candles, int period = 10, double multiplier = 3.0);

#endif // SUPERTREND_STRATEGY_H