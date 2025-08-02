#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "data_types.h"
#include "macd_strategy.h"
#include "rsi_strategy.h"
#include "supertrend_strategy.h"

namespace py = pybind11;

// Convert Python list of dictionaries to C++ vector of Candle
std::vector<Candle> convert_to_candles(const py::list& data) {
    std::vector<Candle> candles;
    for (const auto& item : data) {
        py::dict candle_dict = item.cast<py::dict>();
        Candle candle;
        candle.date = candle_dict["date"].cast<std::string>();
        candle.open = candle_dict["open"].cast<double>();
        candle.high = candle_dict["high"].cast<double>();
        candle.low = candle_dict["low"].cast<double>();
        candle.close = candle_dict["close"].cast<double>();
        candle.volume = candle_dict["volume"].cast<double>();
        candles.push_back(candle);
    }
    return candles;
}

// Convert C++ vector of Signal to Python list
py::list convert_signals_to_python(const std::vector<Signal>& signals) {
    py::list result;
    for (const auto& signal : signals) {
        switch (signal) {
            case Signal::BUY:
                result.append("BUY");
                break;
            case Signal::SELL:
                result.append("SELL");
                break;
            case Signal::HOLD:
                result.append("HOLD");
                break;
        }
    }
    return result;
}

// Python wrapper functions
py::list macd_strategy_py(const py::list& data) {
    std::vector<Candle> candles = convert_to_candles(data);
    std::vector<Signal> signals = macd_signals(candles);
    return convert_signals_to_python(signals);
}

py::list rsi_strategy_py(const py::list& data, int period = 14, double overbought = 70.0, double oversold = 30.0) {
    std::vector<Candle> candles = convert_to_candles(data);
    std::vector<Signal> signals = rsi_signals(candles, period, overbought, oversold);
    return convert_signals_to_python(signals);
}

py::list supertrend_strategy_py(const py::list& data, int period = 10, double multiplier = 3.0) {
    std::vector<Candle> candles = convert_to_candles(data);
    std::vector<Signal> signals = supertrend_signals(candles, period, multiplier);
    return convert_signals_to_python(signals);
}

PYBIND11_MODULE(strategies, m) {
    m.doc() = "Stock trading strategies implemented in C++ with pybind11"; // Optional module docstring
    
    // Expose the strategy functions
    m.def("macd_strategy", &macd_strategy_py, 
          "MACD strategy - generates BUY/SELL signals based on MACD crossover",
          py::arg("data"));
    
    m.def("rsi_strategy", &rsi_strategy_py, 
          "RSI strategy - generates BUY/SELL signals based on RSI overbought/oversold levels",
          py::arg("data"), py::arg("period") = 14, 
          py::arg("overbought") = 70.0, py::arg("oversold") = 30.0);
    
    m.def("supertrend_strategy", &supertrend_strategy_py, 
          "Supertrend strategy - generates BUY/SELL signals based on Supertrend indicator",
          py::arg("data"), py::arg("period") = 10, py::arg("multiplier") = 3.0);
} 