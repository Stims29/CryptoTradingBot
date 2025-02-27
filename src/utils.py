#!/usr/bin/env python
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def format_duration(td: timedelta) -> str:
    """Formate une durée en format lisible."""
    try:
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
    except Exception:
        return str(td)

def calculate_trade_metrics(trades: List[Dict]) -> Dict:
    """Calcule les métriques détaillées sur une série de trades."""
    try:
        if not trades:
            return {}
            
        profits = [t['pnl'] for t in trades]
        holding_times = [(t['close_time'] - t['open_time']).total_seconds() for t in trades]
        
        # Métriques de base
        total_trades = len(trades)
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_profit': sum(profits),
            'avg_profit': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'profit_factor': (
                sum([p for p in profits if p > 0]) / abs(sum([p for p in profits if p < 0]))
                if sum([p for p in profits if p < 0]) != 0 else float('inf')
            ),
            'avg_holding_time': np.mean(holding_times) if holding_times else 0,
            'max_holding_time': max(holding_times) if holding_times else 0,
            'min_holding_time': min(holding_times) if holding_times else 0
        }
        
        # Streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for pnl in profits:
            if pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = min(max_loss_streak, current_streak)
        
        metrics.update({
            'max_win_streak': max_win_streak,
            'max_loss_streak': abs(max_loss_streak)
        })
        
        # Drawdown
        running_pnl = np.cumsum(profits)
        peak = 0
        drawdowns = []
        current_dd = 0
        
        for pnl in running_pnl:
            if pnl > peak:
                peak = pnl
                current_dd = 0
            else:
                current_dd = pnl - peak
            drawdowns.append(current_dd)
        
        metrics['max_drawdown'] = abs(min(drawdowns)) if drawdowns else 0
        metrics['avg_drawdown'] = abs(np.mean([d for d in drawdowns if d < 0])) if drawdowns else 0
        
        return metrics
        
    except Exception as e:
        logging.error(f"Erreur calcul métriques trades: {e}")
        return {}

def analyze_market_volatility(prices: pd.Series, window: int = 20) -> Tuple[float, float]:
    """Analyse la volatilité d'un marché."""
    try:
        returns = prices.pct_change()
        rolling_std = returns.rolling(window=window).std()
        
        current_vol = rolling_std.iloc[-1] * 100  # En pourcentage
        avg_vol = rolling_std.mean() * 100
        
        return current_vol, avg_vol
        
    except Exception as e:
        logging.error(f"Erreur calcul volatilité: {e}")
        return 0.0, 0.0

def detect_market_condition(data: pd.DataFrame, params: Dict) -> str:
    """Détecte les conditions de marché actuelles."""
    try:
        if data.empty:
            return 'UNKNOWN'
            
        # Calcul indicateurs
        close = data['close']
        volume = data['volume']
        
        # Volatilité
        current_vol, avg_vol = analyze_market_volatility(close)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        # Volume
        volume_sma = volume.rolling(window=20).mean()
        volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # Tendance
        ema_short = close.ewm(span=8).mean()
        ema_long = close.ewm(span=21).mean()
        trend_strength = abs((ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1])
        
        # Classification conditions
        if vol_ratio > 1.5 and volume_ratio > 1.5:
            if trend_strength > 0.002:  # 0.2%
                return 'TRENDING_VOLATILE'
            return 'VOLATILE'
            
        if trend_strength > 0.002:
            if volume_ratio > 1.2:
                return 'TRENDING_STRONG'
            return 'TRENDING_WEAK'
            
        if vol_ratio < 0.5:
            return 'RANGING_TIGHT'
            
        return 'RANGING_NORMAL'
        
    except Exception as e:
        logging.error(f"Erreur détection conditions: {e}")
        return 'UNKNOWN'

def calculate_optimal_position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    min_size: float = 0.1,
    max_size: float = None
) -> float:
    """Calcule la taille optimale d'une position."""
    try:
        # Calcul taille de base
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        
        # Application limites
        position_size = max(position_size, min_size)
        if max_size:
            position_size = min(position_size, max_size)
            
        return round(position_size, 2)
        
    except Exception as e:
        logging.error(f"Erreur calcul taille position: {e}")
        return 0.0

def format_trade_log(trade: Dict) -> str:
    """Formate les détails d'un trade pour le logging."""
    try:
        return (
            f"{trade['symbol']} {trade['side']} - "
            f"Size: {trade['size']:.2f}€, "
            f"Entry: {trade['entry_price']:.8f}, "
            f"Exit: {trade['exit_price']:.8f}, "
            f"P&L: {trade['pnl']:.2f}€ ({trade['pnl_pct']:.2f}%), "
            f"Time: {format_duration(trade['holding_time'])}"
        )
    except Exception:
        return str(trade)

def load_trade_history(filepath: str) -> pd.DataFrame:
    """Charge et prépare l'historique des trades."""
    try:
        df = pd.read_csv(filepath)
        
        # Conversion timestamps
        for col in ['open_time', 'close_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        # Calcul métriques additionnelles
        if 'pnl' in df.columns and 'size' in df.columns:
            df['roi'] = df['pnl'] / df['size'] * 100
            
        if 'open_time' in df.columns and 'close_time' in df.columns:
            df['holding_time'] = (df['close_time'] - df['open_time']).dt.total_seconds()
            
        return df
        
    except Exception as e:
        logging.error(f"Erreur chargement historique: {e}")
        return pd.DataFrame()