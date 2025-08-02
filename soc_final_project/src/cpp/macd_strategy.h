#ifndef MACD_STRATEGY_H
#define MACD_STRATEGY_H

#include <vector>
#include "data_types.h"

std::vector<Signal> macd_signals(const std::vector<Candle>& candles);

#endif // MACD_STRATEGY_H