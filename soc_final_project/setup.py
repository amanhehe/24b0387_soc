from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "strategies",
        ["src/cpp/binding.cpp",
         "src/cpp/macd_strategy.cpp",
         "src/cpp/rsi_strategy.cpp", 
         "src/cpp/supertrend_strategy.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=11,
    ),
]

setup(
    name="stock_strategies",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
) 