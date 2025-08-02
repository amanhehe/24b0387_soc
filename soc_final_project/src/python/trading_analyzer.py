#!/usr/bin/env python3
"""
Trading Performance Analyzer
Calculates performance metrics for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class TradingAnalyzer:
    """Analyzes trading strategy performance"""
    
    def __init__(self, stock_data: pd.DataFrame, strategy_signals: Dict[str, List[str]]):
        """
        Initialize the analyzer
        
        Args:
            stock_data: DataFrame with OHLCV data
            strategy_signals: Dict mapping strategy names to signal lists
        """
        self.stock_data = stock_data
        self.strategy_signals = strategy_signals
        self.results = {}
    
    def calculate_trades(self, signals: List[str]) -> List[Dict]:
        """
        Extract trades from signals
        
        Args:
            signals: List of BUY/SELL/HOLD signals
            
        Returns:
            List of trade dictionaries with entry/exit info
        """
        trades = []
        in_position = False
        entry_price = 0
        entry_date = ""
        
        for i, signal in enumerate(signals):
            if signal == "BUY" and not in_position:
                # Enter position
                in_position = True
                entry_price = self.stock_data.iloc[i]['close']
                entry_date = self.stock_data.iloc[i]['date']
                
            elif signal == "SELL" and in_position:
                # Exit position
                exit_price = self.stock_data.iloc[i]['close']
                exit_date = self.stock_data.iloc[i]['date']
                
                # Calculate trade metrics
                trade_return = (exit_price - entry_price) / entry_price
                trade_duration = i - self.stock_data[self.stock_data['date'] == entry_date].index[0]
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'duration': trade_duration,
                    'profit': exit_price - entry_price
                })
                
                in_position = False
        
        return trades
    
    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from trades
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'success_rate': 0.0,
                'per_trade_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_return': 0.0,
                'avg_trade_duration': 0
            }
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['return'] > 0])
        losing_trades = len([t for t in trades if t['return'] <= 0])
        
        success_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        per_trade_return = np.mean([t['return'] for t in trades])
        total_return = sum([t['return'] for t in trades])
        avg_trade_duration = np.mean([t['duration'] for t in trades])
        
        return {
            'success_rate': success_rate,
            'per_trade_return': per_trade_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_return': total_return,
            'avg_trade_duration': avg_trade_duration
        }
    
    def analyze_strategy(self, strategy_name: str, signals: List[str]) -> Dict:
        """
        Analyze a single strategy
        
        Args:
            strategy_name: Name of the strategy
            signals: List of signals for the strategy
            
        Returns:
            Dictionary with strategy performance metrics
        """
        # Extract trades
        trades = self.calculate_trades(signals)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(trades)
        
        # Add strategy name
        metrics['strategy'] = strategy_name
        
        return metrics
    
    def analyze_all_strategies(self) -> Dict[str, Dict]:
        """
        Analyze all strategies
        
        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        results = {}
        
        for strategy_name, signals in self.strategy_signals.items():
            results[strategy_name] = self.analyze_strategy(strategy_name, signals)
        
        return results 