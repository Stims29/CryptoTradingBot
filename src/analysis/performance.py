#!/usr/bin/env python
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..exceptions import ValidationError
from ..logger import setup_logging

class PerformanceAnalyzer:
    def __init__(self):
        """Initialise l'analyseur de performance."""
        self.logger = setup_logging(__name__)

    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyse les trades."""
        try:
            if not trades:
                return {}
                
            # Conversion en DataFrame
            df = pd.DataFrame(trades)
            df['duration'] = pd.to_timedelta(df['duration'], unit='s')
            
            # Métriques de base
            total_trades = len(trades)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] <= 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = df['pnl'].sum()
            avg_pnl = df['pnl'].mean()
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': float(win_rate),
                'total_pnl': float(total_pnl),
                'avg_pnl': float(avg_pnl),
                'max_profit': float(df['pnl'].max()),
                'max_loss': float(df['pnl'].min()),
                'avg_win': float(df[df['pnl'] > 0]['pnl'].mean()),
                'avg_loss': float(df[df['pnl'] <= 0]['pnl'].mean()),
                'profit_factor': float(
                    abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] <= 0]['pnl'].sum())
                    if df[df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
                ),
                'avg_duration': str(df['duration'].mean()),
                'max_duration': str(df['duration'].max()),
                'min_duration': str(df['duration'].min())
            }
            
            # Analyse par catégorie
            category_metrics = {}
            for category in df['symbol'].unique():
                cat_df = df[df['symbol'] == category]
                cat_wins = len(cat_df[cat_df['pnl'] > 0])
                
                category_metrics[category] = {
                    'trades': len(cat_df),
                    'win_rate': float(cat_wins / len(cat_df) * 100),
                    'pnl': float(cat_df['pnl'].sum()),
                    'avg_pnl': float(cat_df['pnl'].mean()),
                    'max_profit': float(cat_df['pnl'].max()),
                    'max_loss': float(cat_df['pnl'].min()),
                    'avg_duration': str(cat_df['duration'].mean())
                }
            
            metrics['category_performance'] = category_metrics
            
            # Analyse temporelle
            df.set_index('open_time', inplace=True)
            hourly_pnl = df.resample('1H')['pnl'].sum()
            
            metrics['hourly_performance'] = {
                'best_hour': int(hourly_pnl.idxmax().hour),
                'worst_hour': int(hourly_pnl.idxmin().hour),
                'avg_hourly_pnl': float(hourly_pnl.mean()),
                'profitable_hours': int((hourly_pnl > 0).sum()),
                'unprofitable_hours': int((hourly_pnl <= 0).sum())
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur analyse trades: {e}")
            return {}

    def calculate_drawdown(self, pnl_series: pd.Series) -> Dict:
        """Calcule les drawdowns."""
        try:
            cumulative = pnl_series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max)
            
            max_dd = drawdown.min()
            max_dd_duration = self._get_drawdown_duration(drawdown)
            
            # Périodes de drawdown
            dd_periods = []
            current_dd = 0
            dd_start = None
            
            for date, dd in drawdown.items():
                if dd < 0 and current_dd == 0:
                    dd_start = date
                    current_dd = dd
                elif dd == 0 and current_dd < 0:
                    dd_periods.append({
                        'start': dd_start,
                        'end': date,
                        'drawdown': float(current_dd),
                        'duration': str(date - dd_start)
                    })
                    current_dd = 0
                elif dd < current_dd:
                    current_dd = dd
            
            return {
                'max_drawdown': float(max_dd),
                'max_drawdown_duration': str(max_dd_duration),
                'avg_drawdown': float(drawdown[drawdown < 0].mean()),
                'drawdown_periods': dd_periods
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul drawdown: {e}")
            return {}

    def _get_drawdown_duration(self, drawdown: pd.Series) -> timedelta:
        """Calcule la durée du drawdown maximal."""
        try:
            # Identification période
            dd = 0
            max_dd = 0
            dd_start = None
            max_duration = timedelta(0)
            
            for date, value in drawdown.items():
                if value < 0 and dd == 0:
                    dd_start = date
                    dd = value
                elif value == 0 and dd < 0:
                    duration = date - dd_start
                    if duration > max_duration:
                        max_duration = duration
                    dd = 0
                elif value < dd:
                    dd = value
                    
            return max_duration
            
        except Exception as e:
            self.logger.error(f"Erreur calcul durée drawdown: {e}")
            return timedelta(0)

    def calculate_risk_metrics(self, trades: List[Dict], capital: float) -> Dict:
        """Calcule les métriques de risque."""
        try:
            df = pd.DataFrame(trades)
            
            # Calcul risque
            avg_risk = df['size'].mean() / capital * 100
            max_risk = df['size'].max() / capital * 100
            
            # Ratio Sharpe simplifié
            returns = df['pnl_pct']
            risk_free_rate = 0.02  # 2% annuel
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            
            excess_returns = returns - daily_rf
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            # Ratio Sortino
            downside_std = returns[returns < 0].std()
            sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
            
            return {
                'avg_risk_per_trade': float(avg_risk),
                'max_risk_per_trade': float(max_risk),
                'sharpe_ratio': float(sharpe),
                'sortino_ratio': float(sortino),
                'max_consecutive_losses': self._get_max_consecutive_losses(df),
                'avg_win_loss_ratio': float(
                    abs(df[df['pnl'] > 0]['pnl'].mean() / df[df['pnl'] <= 0]['pnl'].mean())
                    if len(df[df['pnl'] <= 0]) > 0 else float('inf')
                )
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul métriques risque: {e}")
            return {}

    def _get_max_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calcule le nombre maximum de pertes consécutives."""
        try:
            consecutive = 0
            max_consecutive = 0
            
            for pnl in df['pnl']:
                if pnl <= 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
                    
            return max_consecutive
            
        except Exception as e:
            self.logger.error(f"Erreur calcul pertes consécutives: {e}")
            return 0

    def get_performance_summary(self, trades: List[Dict], capital: float) -> Dict:
        """Génère un résumé complet des performances."""
        try:
            trade_metrics = self.analyze_trades(trades)
            
            if not trade_metrics:
                return {}
                
            # Conversion trades en séries temporelles
            df = pd.DataFrame(trades)
            df.set_index('open_time', inplace=True)
            
            # Calcul drawdown
            dd_metrics = self.calculate_drawdown(df['pnl'])
            
            # Calcul risque
            risk_metrics = self.calculate_risk_metrics(trades, capital)
            
            return {
                'trades': trade_metrics,
                'drawdown': dd_metrics,
                'risk': risk_metrics,
                'summary': {
                    'total_trades': trade_metrics['total_trades'],
                    'win_rate': trade_metrics['win_rate'],
                    'total_pnl': trade_metrics['total_pnl'],
                    'max_drawdown': dd_metrics['max_drawdown'],
                    'sharpe_ratio': risk_metrics['sharpe_ratio'],
                    'profit_factor': trade_metrics['profit_factor']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur génération résumé: {e}")
            return {}