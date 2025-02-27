#!/usr/bin/env python
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..exceptions import ValidationError
from ..logger import setup_logging

class MarketMetrics:
    def __init__(self):
        """Initialise le calculateur de métriques."""
        self.logger = setup_logging(__name__)

    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Calcule les métriques de volatilité."""
        try:
            # Calcul rendements
            returns = data['close'].pct_change()
            
            # Volatilité classique
            volatility = returns.std() * np.sqrt(252) * 100  # Annualisée en %
            
            # Volatilité par fenêtre
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            current_vol = rolling_vol.iloc[-1]
            
            # Volatilité haute/basse
            high_vol = rolling_vol.max()
            low_vol = rolling_vol[rolling_vol > 0].min()
            
            return {
                'current_volatility': float(current_vol),
                'average_volatility': float(volatility),
                'high_volatility': float(high_vol),
                'low_volatility': float(low_vol),
                'volatility_trend': 'UP' if current_vol > volatility else 'DOWN'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul volatilité: {e}")
            return {}

    def calculate_trend_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcule les métriques de tendance."""
        try:
            # EMAs
            ema_short = data['close'].ewm(span=8).mean()
            ema_medium = data['close'].ewm(span=21).mean()
            ema_long = data['close'].ewm(span=55).mean()
            
            # Force tendance
            trend_strength = abs(
                (ema_short.iloc[-1] - ema_long.iloc[-1]) / 
                ema_long.iloc[-1]
            ) * 100
            
            # Direction tendance
            trend_direction = (
                'UP' if ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]
                else 'DOWN' if ema_short.iloc[-1] < ema_medium.iloc[-1] < ema_long.iloc[-1]
                else 'NEUTRAL'
            )
            
            # Momentum
            momentum = data['close'].diff(5) / data['close'].shift(5) * 100
            current_momentum = momentum.iloc[-1]
            
            return {
                'trend_strength': float(trend_strength),
                'trend_direction': trend_direction,
                'current_momentum': float(current_momentum),
                'ema_alignment': trend_direction != 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul tendance: {e}")
            return {}

    def calculate_volume_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcule les métriques de volume."""
        try:
            # Volume moyen
            volume_sma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            
            # Ratio volume
            volume_ratio = current_volume / volume_sma.iloc[-1]
            
            # Volume par tendance
            up_volume = data[data['close'] > data['close'].shift(1)]['volume'].mean()
            down_volume = data[data['close'] < data['close'].shift(1)]['volume'].mean()
            
            # Volume trend
            volume_trend = (
                'UP' if current_volume > volume_sma.iloc[-1] * 1.5
                else 'DOWN' if current_volume < volume_sma.iloc[-1] * 0.5
                else 'NORMAL'
            )
            
            return {
                'current_volume': float(current_volume),
                'average_volume': float(volume_sma.iloc[-1]),
                'volume_ratio': float(volume_ratio),
                'up_down_ratio': float(up_volume / down_volume if down_volume > 0 else 1),
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul volume: {e}")
            return {}

    def calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calcule les niveaux de support/résistance."""
        try:
            # Pivots
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # Distance aux niveaux
            current_price = data['close'].iloc[-1]
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'resistance_2': float(r2),
                'resistance_3': float(r3),
                'support_1': float(s1),
                'support_2': float(s2),
                'support_3': float(s3),
                'nearest_support': float(max(s for s in [s1, s2, s3] if s < current_price)),
                'nearest_resistance': float(min(r for r in [r1, r2, r3] if r > current_price))
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul supports/résistances: {e}")
            return {}

    def calculate_market_condition(self, data: pd.DataFrame) -> str:
        """Détermine les conditions de marché actuelles."""
        try:
            # Calcul métriques
            volatility = self.calculate_volatility(data)
            trend = self.calculate_trend_metrics(data)
            volume = self.calculate_volume_metrics(data)
            
            # Classification conditions
            if trend['trend_direction'] != 'NEUTRAL':
                if volatility['current_volatility'] > volatility['average_volatility']:
                    if volume['volume_ratio'] > 1.5:
                        return 'TRENDING_VOLATILE'
                    return 'TRENDING'
                    
                if volume['volume_ratio'] > 1.5:
                    return 'TRENDING_STRONG'
                return 'TRENDING_WEAK'
                
            if volatility['current_volatility'] < volatility['low_volatility'] * 1.5:
                return 'RANGING_TIGHT'
                
            if volatility['current_volatility'] > volatility['high_volatility'] * 0.8:
                return 'RANGING_WIDE'
                
            return 'RANGING_NORMAL'
            
        except Exception as e:
            self.logger.error(f"Erreur détermination conditions: {e}")
            return 'UNKNOWN'

    def get_all_metrics(self, data: pd.DataFrame) -> Dict:
        """Calcule toutes les métriques."""
        try:
            return {
                'volatility': self.calculate_volatility(data),
                'trend': self.calculate_trend_metrics(data),
                'volume': self.calculate_volume_metrics(data),
                'levels': self.calculate_support_resistance(data),
                'market_condition': self.calculate_market_condition(data)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul métriques: {e}")
            return {}