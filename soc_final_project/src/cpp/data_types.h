#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <string>

struct Candle {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

enum class Signal {
    BUY,
    SELL,
    HOLD
};

#endif // DATA_TYPES_H