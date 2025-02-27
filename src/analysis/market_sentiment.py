#!/usr/bin/env python
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..exceptions import ValidationError
from ..logger import setup_logging

class MarketSentimentAnalyzer:
    def __init__(self):
        """Initialise l'analyseur de sentiment."""
        self.logger = setup_logging(__name__)
        
        # Configuration des indicateurs par catégorie
        self.indicators_config = {
            'MAJOR': {
                'rsi_periods': [9, 14, 21],   # Périodes RSI
                'ema_periods': [9, 21, 50],   # EMAs 
                'vwap_period': 14,            # VWAP
                'bb_period': 20,              # Bollinger
                'bb_std': 2.0,                # Écart-type BB
                'volume_ma': 20,              # MA Volume
                'momentum_period': 10,        # Momentum
                'support_levels': 3,          # Niveaux support
                'resistance_levels': 3         # Niveaux résistance
            },
            'ALTCOINS': {
                'rsi_periods': [7, 14, 21],
                'ema_periods': [7, 21, 50],
                'vwap_period': 14,
                'bb_period': 20,
                'bb_std': 2.2,
                'volume_ma': 20,
                'momentum_period': 8,
                'support_levels': 3,
                'resistance_levels': 3
            },
            'DEFI': {
                'rsi_periods': [6, 14, 21],
                'ema_periods': [6, 21, 50],
                'vwap_period': 14,
                'bb_period': 20,
                'bb_std': 2.5,
                'volume_ma': 20,
                'momentum_period': 6,
                'support_levels': 2,
                'resistance_levels': 2
            },
            'NFT': {
                'rsi_periods': [6, 14, 21],
                'ema_periods': [6, 21, 50],
                'vwap_period': 14,
                'bb_period': 20,
                'bb_std': 2.5,
                'volume_ma': 20,
                'momentum_period': 6,
                'support_levels': 2,
                'resistance_levels': 2
            },
            'BTC_PAIRS': {
                'rsi_periods': [12, 14, 21],
                'ema_periods': [12, 21, 50],
                'vwap_period': 14,
                'bb_period': 20,
                'bb_std': 1.8,
                'volume_ma': 20,
                'momentum_period': 12,
                'support_levels': 4,
                'resistance_levels': 4
            }
        }
        
        # Cache pour éviter recalculs fréquents
        self.tech_cache = {}
        self.sentiment_cache = {}
        self.support_resistance_cache = {}
        self.cache_duration = timedelta(milliseconds=250)

    async def analyze_technical_indicators(
        self,
        symbol: str,
        data: pd.DataFrame = None
    ) -> Dict:
        """Analyse les indicateurs techniques."""
        try:
            # Log début analyse
            self.logger.debug(f"\nAnalyse technique {symbol}")

            # Vérification cache
            cache_key = f"{symbol}_tech"
            if cache_key in self.tech_cache:
                cache_data = self.tech_cache[cache_key]
                if datetime.now() - cache_data['timestamp'] < self.cache_duration:
                    return cache_data['data']

            if data is None or data.empty or len(data) < 50:
                self.logger.warning(f"Données insuffisantes pour {symbol}")
                return {'score': 0}

            try:
                # Détermination configuration
                category = 'MAJOR' if '/USDT' in symbol else 'ALTCOINS'
                config = self.indicators_config.get(category)
                
                # Calcul des indicateurs
                technical_scores = self._calculate_technical_scores(data, config)
                momentum_scores = self._calculate_momentum_scores(data, config)
                volume_scores = self._calculate_volume_scores(data, config)
                sr_scores = self._calculate_support_resistance_scores(data, config)
                
                # Pondération des scores
                weights = {
                    'technical': 0.35,
                    'momentum': 0.25,
                    'volume': 0.20,
                    'sr': 0.20
                }
                
                composite_score = (
                    technical_scores['score'] * weights['technical'] +
                    momentum_scores['score'] * weights['momentum'] +
                    volume_scores['score'] * weights['volume'] +
                    sr_scores['score'] * weights['sr']
                )
                
                # Résultats
                result = {
                    'score': float(np.clip(composite_score, -1, 1)),
                    'rsi': technical_scores['rsi'],
                    'ema_signals': technical_scores['ema_signals'],
                    'bb_position': technical_scores['bb_position'],
                    'vwap_distance': technical_scores['vwap_distance'],
                    'momentum': momentum_scores['momentum'],
                    'volume_trend': volume_scores['volume_trend'],
                    'support_resistance': sr_scores['levels'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Log détaillé
                self.logger.debug(
                    f"Indicateurs {symbol}:\n"
                    f"RSI: {result['rsi']:.1f}\n"
                    f"EMA Signals: {result['ema_signals']}\n"
                    f"BB Position: {result['bb_position']:.2f}\n"
                    f"VWAP Distance: {result['vwap_distance']:.2f}%\n"
                    f"Score final: {result['score']:.3f}"
                )
                
                # Mise en cache
                self.tech_cache[cache_key] = {
                    'timestamp': datetime.now(),
                    'data': result
                }
                
                return result

            except Exception as e:
                self.logger.error(f"Erreur calcul indicateurs pour {symbol}: {e}")
                return {'score': 0}

        except Exception as e:
            self.logger.error(f"Erreur analyse technique pour {symbol}: {e}")
            return {'score': 0}

    def _calculate_technical_scores(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Calcule les scores des indicateurs techniques."""
        try:
            close = data['close']
            
            # RSI multi-périodes
            rsi_values = []
            for period in config['rsi_periods']:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).ewm(span=period).mean()
                loss = -delta.where(delta < 0, 0).ewm(span=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi.iloc[-1])
            
            rsi_avg = np.mean(rsi_values)
            rsi_score = (50 - rsi_avg) / 50  # -1 à 1
            
            # EMAs
            ema_signals = {}
            for period in config['ema_periods']:
                ema = close.ewm(span=period).mean()
                ema_signals[f'ema_{period}'] = close.iloc[-1] > ema.iloc[-1]
            
            ema_score = sum(1 if sig else -1 for sig in ema_signals.values()) / len(ema_signals)
            
            # Bollinger Bands
            bb_sma = close.rolling(window=config['bb_period']).mean()
            bb_std = close.rolling(window=config['bb_period']).std()
            bb_upper = bb_sma + config['bb_std'] * bb_std
            bb_lower = bb_sma - config['bb_std'] * bb_std
            
            bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            bb_score = 1 - 2 * bb_position  # 1 près du bas, -1 près du haut
            
            # VWAP
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(window=config['vwap_period']).sum() / \
                   data['volume'].rolling(window=config['vwap_period']).sum()
            
            vwap_distance = (close.iloc[-1] / vwap.iloc[-1] - 1) * 100
            vwap_score = -np.clip(vwap_distance / 2, -1, 1)  # -1 si au-dessus, 1 si en-dessous
            
            # Score composite
            tech_score = np.mean([
                rsi_score * 1.2,    # Plus de poids au RSI
                ema_score,
                bb_score,
                vwap_score
            ])
            
            return {
                'score': float(tech_score),
                'rsi': float(rsi_avg),
                'ema_signals': ema_signals,
                'bb_position': float(bb_position),
                'vwap_distance': float(vwap_distance)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul scores techniques: {e}")
            return {'score': 0, 'rsi': 50, 'ema_signals': {}, 'bb_position': 0.5, 'vwap_distance': 0}

    def _calculate_momentum_scores(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Calcule les scores de momentum."""
        try:
            close = data['close']
            period = config['momentum_period']
            
            # ROC (Rate of Change)
            roc = close.pct_change(period)
            
            # Accélération du momentum
            roc_sma = roc.rolling(window=period).mean()
            momentum = roc.iloc[-1]
            acceleration = (roc.iloc[-1] - roc_sma.iloc[-1])
            
            # Score composite
            momentum_score = np.clip(
                momentum * 5 + np.sign(acceleration) * 0.5,
                -1, 1
            )
            
            return {
                'score': float(momentum_score),
                'momentum': float(momentum),
                'acceleration': float(acceleration)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul scores momentum: {e}")
            return {'score': 0, 'momentum': 0, 'acceleration': 0}

    def _calculate_volume_scores(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Calcule les scores basés sur le volume."""
        try:
            volume = data['volume']
            close = data['close']
            period = config['volume_ma']
            
            # Volume moyen
            volume_sma = volume.rolling(window=period).mean()
            relative_volume = volume / volume_sma
            
            # Volume par direction
            up_volume = volume.where(close > close.shift(1), 0)
            down_volume = volume.where(close <= close.shift(1), 0)
            
            up_volume_sma = up_volume.rolling(window=period).mean()
            down_volume_sma = down_volume.rolling(window=period).mean()
            
            # Ratio acheteurs/vendeurs
            volume_ratio = (
                (up_volume_sma.iloc[-1] / down_volume_sma.iloc[-1]) 
                if down_volume_sma.iloc[-1] > 0 else 1
            )
            
            # Score composite
            volume_score = np.clip(
                (np.log(volume_ratio) + (relative_volume.iloc[-1] - 1)) / 2,
                -1, 1
            )
            
            return {
                'score': float(volume_score),
                'volume_trend': float(volume_ratio),
                'relative_volume': float(relative_volume.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul scores volume: {e}")
            return {'score': 0, 'volume_trend': 1, 'relative_volume': 1}

    def _calculate_support_resistance_scores(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Calcule les niveaux de support/résistance."""
        try:
            close = data['close'].iloc[-1]
            high = data['high']
            low = data['low']
            
            # Identification pivots
            n_levels = max(config['support_levels'], config['resistance_levels'])
            window = 5
            
            pivot_highs = []
            pivot_lows = []
            
            for i in range(window, len(data) - window):
                if all(high.iloc[i] > high.iloc[i-window:i]) and \
                   all(high.iloc[i] > high.iloc[i+1:i+window+1]):
                    pivot_highs.append(high.iloc[i])
                
                if all(low.iloc[i] < low.iloc[i-window:i]) and \
                   all(low.iloc[i] < low.iloc[i+1:i+window+1]):
                    pivot_lows.append(low.iloc[i])
            
            # Sélection niveaux les plus proches
            pivot_highs = sorted(pivot_highs)[-n_levels:]
            pivot_lows = sorted(pivot_lows)[:n_levels]
            
            # Distance aux niveaux
            if pivot_highs:
                nearest_resistance = min(p for p in pivot_highs if p > close)
                resistance_distance = (nearest_resistance / close - 1) * 100
            else:
                resistance_distance = 100
                
            if pivot_lows:
                nearest_support = max(p for p in pivot_lows if p < close)
                support_distance = (close / nearest_support - 1) * 100
            else:
                support_distance = 100
            
            # Score basé sur la position relative
            if support_distance == 0 or resistance_distance == 0:
                sr_score = 0
            else:
                position = support_distance / (support_distance + resistance_distance)
                sr_score = 2 * (0.5 - position)  # -1 près résistance, 1 près support
            
            return {
                'score': float(sr_score),
                'levels': {
                    'resistances': [float(p) for p in pivot_highs],
                    'supports': [float(p) for p in pivot_lows]
                },
                'distances': {
                    'resistance': float(resistance_distance),
                    'support': float(support_distance)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul niveaux S/R: {e}")
            return {
                'score': 0,
                'levels': {'resistances': [], 'supports': []},
                'distances': {'resistance': 0, 'support': 0}
            }

    async def analyze_news_sentiment(self, symbol: str) -> Dict:
        """Simule un sentiment basé sur la catégorie et les volumes."""
        try:
            # Vérification cache
            cache_key = f"{symbol}_sentiment"
            if cache_key in self.sentiment_cache:
                cache_data = self.sentiment_cache[cache_key]
                if datetime.now() - cache_data['timestamp'] < self.cache_duration:
                    return cache_data['data']
            
            # Simulation sentiment selon catégorie
            if '/USDT' in symbol:
                if 'BTC' in symbol or 'ETH' in symbol:
                    # Plus stable pour les majors
                    base_sentiment = np.random.normal(0, 0.02)
                    impact = np.random.uniform(0.8, 1.2)
                else:
                    # Plus volatile pour les alts
                    base_sentiment = np.random.normal(0, 0.04)
                    impact = np.random.uniform(0.6, 1.4)
            else:
                # Paires BTC très stables
                base_sentiment = np.random.normal(0, 0.01)
                impact = np.random.uniform(0.9, 1.1)

            # Ajustement dynamique
            if symbol in self.tech_cache:
                tech_data = self.tech_cache[symbol]['data']
                
                # Influence du momentum
                if 'momentum' in tech_data:
                    base_sentiment += np.sign(tech_data['momentum']) * 0.01
                
                # Influence du volume
                if 'volume_trend' in tech_data:
                    impact *= tech_data['volume_trend']
            
            result = {
                'score': float(np.clip(base_sentiment, -1, 1)),
                'impact': float(np.clip(impact, 0.5, 2.0)),
                'volume_profile': self._get_volume_profile(symbol),
                'timestamp': datetime.now().isoformat()
            }
            
            # Mise en cache
            self.sentiment_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur analyse sentiment pour {symbol}: {e}")
            return {'score': 0, 'impact': 1.0, 'volume_profile': {}}

    def _get_volume_profile(self, symbol: str) -> Dict:
        """Calcule le profil de volume."""
        try:
            if symbol not in self.tech_cache:
                return {}
                
            tech_data = self.tech_cache[symbol]['data']
            
            return {
                'trend': tech_data.get('volume_trend', 1.0),
                'relative_strength': tech_data.get('relative_volume', 1.0),
                'accumulation': tech_data.get('score', 0) > 0.5,
                'distribution': tech_data.get('score', 0) < -0.5
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul profil volume: {e}")
            return {}

    def analyze_market_structure(
        self,
        symbol: str,
        data: pd.DataFrame = None
    ) -> Dict:
        """Analyse la structure du marché."""
        try:
            if symbol not in self.tech_cache:
                return {}
                
            tech_data = self.tech_cache[symbol]['data']
            sr_levels = tech_data.get('support_resistance', {}).get('levels', {})
            
            market_structure = {
                'trend': {
                    'direction': np.sign(tech_data.get('momentum', 0)),
                    'strength': abs(tech_data.get('momentum', 0)),
                    'maturity': self._get_trend_maturity(tech_data)
                },
                'levels': {
                    'supports': sr_levels.get('supports', []),
                    'resistances': sr_levels.get('resistances', []),
                    'key_level': self._get_key_level(tech_data)
                },
                'volatility': {
                    'current': tech_data.get('bb_position', 0.5),
                    'trend': self._get_volatility_trend(tech_data)
                }
            }
            
            return market_structure
            
        except Exception as e:
            self.logger.error(f"Erreur analyse structure: {e}")
            return {}

    def _get_trend_maturity(self, tech_data: Dict) -> float:
        """Évalue la maturité de la tendance."""
        try:
            momentum = abs(tech_data.get('momentum', 0))
            rsi = tech_data.get('rsi', 50)
            
            if momentum > 0.7 and (rsi > 70 or rsi < 30):
                return 1.0  # Tendance mature
            elif momentum > 0.3:
                return 0.5  # Tendance développement
            else:
                return 0.0  # Pas de tendance claire
                
        except Exception:
            return 0.0

    def _get_key_level(self, tech_data: Dict) -> Optional[float]:
        """Identifie le niveau clé le plus proche."""
        try:
            sr_data = tech_data.get('support_resistance', {})
            distances = sr_data.get('distances', {})
            
            if distances.get('support', 100) < distances.get('resistance', 100):
                return sr_data.get('levels', {}).get('supports', [None])[-1]
            else:
                return sr_data.get('levels', {}).get('resistances', [None])[0]
                
        except Exception:
            return None

    def _get_volatility_trend(self, tech_data: Dict) -> int:
        """Détermine la tendance de la volatilité."""
        try:
            bb_pos = tech_data.get('bb_position', 0.5)
            momentum = tech_data.get('momentum', 0)
            
            if bb_pos > 0.8 and momentum > 0:
                return 1  # Volatilité croissante
            elif bb_pos < 0.2 and momentum < 0:
                return -1  # Volatilité décroissante
            else:
                return 0  # Volatilité stable
                
        except Exception:
            return 0

    async def close(self):
        """Ferme proprement l'analyseur."""
        try:
            # Nettoyage des caches
            self.tech_cache.clear()
            self.sentiment_cache.clear()
            self.support_resistance_cache.clear()
            
            self.logger.info("MarketSentimentAnalyzer fermé proprement")
            
        except Exception as e:
            self.logger.error(f"Erreur fermeture analyzer: {e}")