#!/usr/bin/env python
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def format_duration(td: timedelta) -> str:
    """Formate une durée en format lisible."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def calculate_trade_metrics(trades: List[Dict]) -> Dict:
    """Calcule les métriques sur une série de trades."""
    if not trades:
        return {}
        
    profits = [t['pnl'] for t in trades]
    holding_times = [(t['close_time'] - t['open_time']).total_seconds() for t in trades]
    
    return {
        'total_trades': len(trades),
        'winning_trades': len([p for p in profits if p > 0]),
        'losing_trades': len([p for p in profits if p <= 0]),
        'win_rate': (len([p for p in profits if p > 0]) / len(trades) * 100),
        'total_profit': sum(profits),
        'avg_profit': np.mean(profits),
        'max_profit': max(profits),
        'max_loss': min(profits),
        'avg_holding_time': np.mean(holding_times)
    }

def detect_market_condition(data: pd.DataFrame) -> str:
    """Détecte les conditions actuelles du marché."""
    returns = data['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)
    volume_sma = data['volume'].rolling(window=20).mean()
    current_volume = data['volume'].iloc[-1]
    
    if volatility > 0.02:  # Haute volatilité
        if current_volume > volume_sma.iloc[-1] * 1.5:
            return 'VOLATILE_HIGH_VOLUME'
        return 'VOLATILE'
    
    if volatility < 0.005:  # Basse volatilité
        return 'RANGING'
        
    return 'NORMAL'