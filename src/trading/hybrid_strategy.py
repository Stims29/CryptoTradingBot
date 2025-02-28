#!/usr/bin/env python
# src/trading/hybrid_strategy.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

class TechnicalIndicators:
    """Calculs optimisés des indicateurs techniques."""

    def calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """
        Calcule le RSI (Relative Strength Index).
        
        Args:
            prices (pd.Series): Série de prix
            periods (int): Nombre de périodes pour le calcul (défaut: 14)
            
        Returns:
            float: Valeur du RSI (0-100)
        """
        try:
            # Conversion en float
            prices = prices.astype(float)
            
            # Calcul des variations de prix
            delta = prices.diff().fillna(0)
            
            # Séparation gains et pertes
            gain = pd.Series(0, index=delta.index, dtype=float)
            loss = pd.Series(0, index=delta.index, dtype=float)
            
            gain[delta > 0] = delta[delta > 0]
            loss[delta < 0] = -delta[delta < 0]
            
            # Calcul moyenne mobile exponentielle des gains et pertes
            avg_gain = gain.ewm(alpha=1/periods, min_periods=periods).mean()
            avg_loss = loss.ewm(alpha=1/periods, min_periods=periods).mean()
            
            # Calcul du RSI
            rs = avg_gain / avg_loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            # Retourne dernière valeur du RSI
            return rsi.iloc[-1]
                
        except Exception as e:
            logging.error(f"Erreur calcul RSI: {e}")
            return 50.0  # Valeur neutre en cas d'erreur

    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """
        Calcule le MACD (Moving Average Convergence Divergence).
        
        Args:
            prices (pd.Series): Série de prix
            fast_period (int): Période courte EMA (défaut: 12)
            slow_period (int): Période longue EMA (défaut: 26)
            signal_period (int): Période signal EMA (défaut: 9)
            
        Returns:
            Dict[str, float]: Dictionnaire contenant 'macd', 'signal' et 'hist'
        """
        try:
            # Conversion en float
            prices = prices.astype(float)
            
            # Calcul des EMA
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            
            # Calcul du MACD
            macd_line = ema_fast - ema_slow
            
            # Calcul de la ligne de signal
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calcul de l'histogramme
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'hist': histogram.iloc[-1]
            }
                
        except Exception as e:
            logging.error(f"Erreur calcul MACD: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'hist': 0.0}

    def calculate_bollinger_bands(self, prices: pd.Series, periods: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """
        Calcule les bandes de Bollinger.
        
        Args:
            prices (pd.Series): Série de prix
            periods (int): Période de calcul (défaut: 20)
            std_dev (float): Nombre d'écarts-types (défaut: 2.0)
            
        Returns:
            Dict[str, float]: Dictionnaire contenant 'upper', 'mid', 'lower' et 'width'
        """
        try:
            # Conversion en float
            prices = prices.astype(float)
            
            # Calcul de la moyenne mobile
            mid = prices.rolling(window=periods).mean()
            
            # Calcul de l'écart-type
            std = prices.rolling(window=periods).std()
            
            # Calcul des bandes
            upper = mid + std_dev * std
            lower = mid - std_dev * std
            
            # Calcul de la largeur
            width = (upper - lower) / mid
            
            return {
                'upper': upper.iloc[-1],
                'mid': mid.iloc[-1],
                'lower': lower.iloc[-1],
                'width': width.iloc[-1]
            }
                
        except Exception as e:
            logging.error(f"Erreur calcul Bollinger Bands: {e}")
            current = prices.iloc[-1] if not prices.empty else 0.0
            return {'upper': current*1.01, 'mid': current, 'lower': current*0.99, 'width': 0.02}

    def calculate_ema(self, prices: pd.Series, periods: int = 20) -> float:
        """Calcule l'EMA (Exponential Moving Average)."""
        try:
            ema = prices.ewm(span=periods, adjust=False).mean()
            return ema.iloc[-1]
        except Exception as e:
            logging.error(f"Erreur calcul EMA: {e}")
            return prices.iloc[-1] if not prices.empty else 0.0

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, periods: int = 14) -> float:
        """Calcule l'ATR (Average True Range)."""
        try:
            # Calcul du True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calcul de l'ATR
            atr = tr.rolling(window=periods).mean()
            
            return atr.iloc[-1]
        except Exception as e:
            logging.error(f"Erreur calcul ATR: {e}")
            return 0.01  # Valeur par défaut

@dataclass
class SignalMetrics:
    """Métriques pour le suivi des signaux."""
    generated: int = 0
    filtered: int = 0
    executed: int = 0 
    rejected: int = 0
    errors: int = 0
    strategy_signals: Dict[str, int] = None
    
    def __post_init__(self):
        self.strategy_signals = {
            'BREAKOUT': 0,
            'MEAN_REVERSION': 0,
            'MOMENTUM': 0,
            'ORDER_BOOK': 0
        }

class HybridStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()
        
        # Configuration optimisée pour scalping
        self.config = {
            'rsi': {
                'period': 6,                  # Période courte pour scalping
                'overbought': 75,             # Seuil de surachat
                'oversold': 25,               # Seuil de survente
                'neutral_zone': {
                    'low': 45,
                    'high': 55
                }
            },
            'macd': {
                'fast': 3,                    # Ultra-rapide pour scalping
                'slow': 7,                    # Réactif pour scalping
                'signal': 2,                  # Réactif pour signaler des retournements
                'min_strength': 0.0001        # Très sensible pour scalping
            },
            'bollinger': {
                'period': 14,                 # Adapté pour scalping
                'std_dev': 1.8,               # Sensible aux variations
                'min_width': 0.001            # Ajusté pour détecter les opportunités
            },
            'scalping': {
                'signal_cooldown': 2,         # 2 secondes entre signaux
                'min_volatility': 0.0002,     # Volatilité minimum acceptable
                'max_volatility': 0.05,       # Volatilité maximum acceptable
                'min_volume': 1000,           # Volume minimum acceptable
                'confidence_threshold': 0.001, # Seuil très bas pour tests
                'max_signals_per_hour': 100   # Limite maximum par heure
            },
            'trend': {
                'ema_fast': 5,                # EMA très rapide
                'ema_slow': 12,               # EMA lente pour confirmation
                'atr_period': 5,              # Période ATR court terme
                'min_trend_strength': 0.0001  # Sensible pour détecter les micro-tendances
            },
            # Ajout de poids pour les stratégies
            'strategy_weights': {
                'MOMENTUM': 0.30,              # Priorité équilibrée
                'MEAN_REVERSION': 0.30,        # Priorité équilibrée
                'BREAKOUT': 0.30,              # Priorité équilibrée
                'ORDER_BOOK': 0.10             # Priorité moindre
            }
        }

        # Flags de contrôle du logging
        self._verbose_logging = False
        self._log_interval = 10
        self._log_counter = 0
        
        # État et suivi
        self.last_signals = {}
        self.signal_counts = {}
        self.metrics = SignalMetrics()
        self.last_reset = datetime.now()
        
        # Cache de calculs
        self.indicator_cache = {}
        self.cache_expiry = timedelta(seconds=5)
        self.last_cache_cleanup = datetime.now()

    def _should_log(self, importance='normal') -> bool:
        """
        Détermine si un événement doit être loggé selon son importance.
    
        Args:
            importance (str): 'critical', 'high', 'normal', ou 'low'
        
        Returns:
            bool: True si l'événement doit être loggé
        """
        try:
            if importance == 'critical':
                return True  # Toujours logger les événements critiques
            
            if importance == 'high':
                return self._verbose_logging or (self._log_counter % 5 == 0) 
            
            if importance == 'normal':
                return self._verbose_logging or (self._log_counter % self._log_interval == 0)
            
            # Low importance
            return self._verbose_logging
            
        finally:
            # Incrémenter le compteur dans tous les cas
            self._log_counter = (self._log_counter + 1) % 1000

    async def generate_signal(self, symbol: str, market_data: pd.DataFrame, tech_analysis: Dict) -> Optional[Dict]:
        """Génère un signal de trading avec correction d'erreur d'unpacking."""
        try:
            self.logger.info(f"[DEBUG] Entrée generate_signal pour {symbol}")
        
            # Vérifications préliminaires sur les données
            if market_data is None or market_data.empty:
                self.logger.warning(f"Données de marché vides pour {symbol}")
                return None
            
            self.logger.info(f"[DEBUG] Market Data Shape: {market_data.shape}")
            self.logger.info(f"[DEBUG] Tech Analysis Keys: {list(tech_analysis.keys())}")
        
            # Calcul des scores pour chaque stratégie
            breakout_score = self._analyze_breakout(market_data, tech_analysis)
            mean_reversion_score = self._analyze_mean_reversion(market_data, tech_analysis)
            momentum_score = self._analyze_momentum(market_data, tech_analysis)
            order_book_score = self._analyze_orderbook(tech_analysis)
        
            # Combinaison des scores
            scores = {
                'BREAKOUT': breakout_score,
                'MEAN_REVERSION': mean_reversion_score,
                'MOMENTUM': momentum_score,
                'ORDER_BOOK': order_book_score
            }
        
            # Stratégie dominante
            dominant_strategy = max(scores, key=scores.get)
            composite_score = scores[dominant_strategy]
        
            # Log des scores
            self.logger.info(f"Scores {symbol}: BREAKOUT={breakout_score:.4f}, MEAN_REVERSION={mean_reversion_score:.4f}, MOMENTUM={momentum_score:.4f}, ORDER_BOOK={order_book_score:.4f}")
            self.logger.info(f"Score composite {symbol}: {composite_score:.4f} ({dominant_strategy})")
        
            # Validation du signal avec seuils
            valid, reason = self._validate_signal(composite_score, tech_analysis, None)
        
            # Vérification supplémentaire de la performance historique de la stratégie
            if valid and not self._validate_strategy_performance(dominant_strategy):
                valid = False
                reason = f"Stratégie {dominant_strategy} a un mauvais historique récent"
        
            if not valid:
                self.logger.info(f"Signal {symbol} rejeté: {reason}")
                return None
                
            # Déterminer l'action (buy/sell)
            action = 'buy' if composite_score > 0 else 'sell'
        
            # Construction du signal
            signal = {
                'symbol': symbol,
                'action': action,
                'strategy': dominant_strategy,
                'strength': abs(composite_score),
                'timestamp': datetime.now(),
                'metrics': {
                    'price': float(market_data['close'].iloc[-1]),
                    'volume': tech_analysis.get('volume_profile', {}).get('volume_24h', 0),
                    'volatility': tech_analysis.get('volatility', 0),
                    'spread': tech_analysis.get('spread', 0.001)
                },
                'indicators': self._calculate_indicators(market_data) if hasattr(self, '_calculate_indicators') else {}
            }
        
            # Log de confirmation
            self.logger.info(f"Signal généré pour {symbol}: {action} - force: {abs(composite_score):.4f}")
        
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur génération signal: {str(e)}")
            return None
        
    def _validate_strategy_performance(self, strategy: str) -> bool:
        """Vérifie si une stratégie a un historique positif récent"""
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {s: {'win': 0, 'loss': 0} for s in ['BREAKOUT', 'MEAN_REVERSION', 'MOMENTUM', 'ORDER_BOOK']}
            return True
        
        # Si la stratégie a plus de pertes que de gains
        if self.strategy_performance[strategy]['loss'] > self.strategy_performance[strategy]['win'] * 2:
            # Exiger un signal beaucoup plus fort
            return False
        
        return True
    
    def update_strategy_performance(self, strategy: str, is_win: bool):
        """Met à jour les statistiques de performance d'une stratégie en mode simulation."""
        # Initialiser le dictionnaire de performance si nécessaire
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {s: {'win': 0, 'loss': 0} for s in ['BREAKOUT', 'MEAN_REVERSION', 'MOMENTUM', 'ORDER_BOOK']}
    
        # Mettre à jour les stats pour cette stratégie
        if strategy in self.strategy_performance:
            if is_win:
                self.strategy_performance[strategy]['win'] += 1
            else:
                self.strategy_performance[strategy]['loss'] += 1
            
        # Log des performances actualisées
        self.logger.info(f"Performance stratégie {strategy} mise à jour: {self.strategy_performance[strategy]}")

    def _should_process_signal(self, symbol: str, now: datetime) -> Tuple[bool, str]:
        """Vérifie si un signal doit être traité avec cooldown optimisé."""
        try:
            # Mode test: accepter tous les signaux pour diagnostiquer la chaîne d'exécution
            # Réinitialisation des compteurs pour éviter le blocage par max_signals_per_hour
            if symbol not in self.signal_counts:
                self.signal_counts[symbol] = {
                    'count': 0,
                    'hour': now.hour
                }
            
            # Réinitialiser le compteur
            self.signal_counts[symbol] = {
                'count': 0,
                'hour': now.hour
            }
            
            # Enregistrer ce signal comme le dernier traité
            self.last_signals[symbol] = now
                
            return True, "Signal autorisé (mode test)"
            
        except Exception as e:
            self.logger.error(f"Erreur vérification signal: {str(e)}")
            return False, f"Erreur: {str(e)}"

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcule les indicateurs techniques pour l'analyse."""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            # RSI
            rsi = self.technical_indicators.calculate_rsi(
                close, 
                periods=self.config['rsi']['period']
            )
            
            # MACD
            macd = self.technical_indicators.calculate_macd(
                close,
                fast_period=self.config['macd']['fast'],
                slow_period=self.config['macd']['slow'],
                signal_period=self.config['macd']['signal']
            )
            
            # Bollinger Bands
            bb = self.technical_indicators.calculate_bollinger_bands(
                close,
                periods=self.config['bollinger']['period'],
                std_dev=self.config['bollinger']['std_dev']
            )
            
            # EMAs
            ema_fast = self.technical_indicators.calculate_ema(
                close,
                periods=self.config['trend']['ema_fast']
            )
            
            ema_slow = self.technical_indicators.calculate_ema(
                close,
                periods=self.config['trend']['ema_slow']
            )
            
            # ATR
            atr = self.technical_indicators.calculate_atr(
                high, low, close,
                periods=self.config['trend']['atr_period']
            )
            
            # Tendance (pente EMA)
            trend = (ema_fast / ema_slow - 1) * 100
            
            return {
                'rsi': rsi,
                'macd': macd['macd'],
                'macd_signal': macd['signal'],
                'macd_hist': macd['hist'],
                'bb_upper': bb['upper'],
                'bb_lower': bb['lower'],
                'bb_mid': bb['mid'],
                'bb_width': bb['width'],
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'atr': atr,
                'trend': trend
            }
                
        except Exception as e:
            self.logger.error(f"Erreur calcul indicateurs: {str(e)}")
            return {}

    def _validate_signal(self, score: float, tech_analysis: Dict, indicators: Dict) -> Tuple[bool, str]:
        """Valide un signal selon le score et les critères définis."""
        try:
            # Validation du score - ajusté pour Test 3A
            if abs(score) > 0.15:  # Réduit de 0.20 à 0.15
                strategy_type = "MEAN_REVERSION" if score < 0 else "MOMENTUM/BREAKOUT"
                return True, f"Signal validé: {strategy_type} (score: {score:.4f})"
        
            return False, "Score trop faible"
    
        except Exception as e:
            self.logger.error(f"Erreur validation signal: {str(e)}")
            return False, f"Erreur validation: {str(e)}"
        
    def _validate_strategy_performance(self, strategy: str) -> bool:
        """Vérifie si une stratégie a un historique positif récent"""
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {s: {'win': 0, 'loss': 0} for s in ['BREAKOUT', 'MEAN_REVERSION', 'MOMENTUM', 'ORDER_BOOK']}
            return True
        
        # Si la stratégie a plus de pertes que de gains
        if self.strategy_performance[strategy]['loss'] > self.strategy_performance[strategy]['win'] * 2:
            # Exiger un signal beaucoup plus fort
            return False
        
        return True

    def _calculate_adaptive_weights(self, data: pd.DataFrame, tech_analysis: Dict) -> Dict[str, float]:
        """
        Calcule des poids adaptatifs pour les différentes stratégies en fonction
        des conditions de marché actuelles.
        
        Args:
            data: DataFrame des données de marché récentes
            tech_analysis: Dictionnaire d'indicateurs techniques
            
        Returns:
            Dictionnaire des poids par stratégie
        """
        try:
            # Extraire les indicateurs pertinents
            volatility = tech_analysis.get('volatility', 0)
            price_levels = tech_analysis.get('price_levels', {})
            volume_profile = tech_analysis.get('volume_profile', {})
            
            # Obtenir prix actuel et niveaux
            current_price = float(data['close'].iloc[-1])
            resistance = price_levels.get('resistance', current_price * 1.01)
            support = price_levels.get('support', current_price * 0.99)
            
            # Calculer distance aux niveaux clés (en pourcentage)
            distance_to_resistance = abs(resistance - current_price) / current_price if resistance else 0.01
            distance_to_support = abs(support - current_price) / current_price if support else 0.01
            
            # Déterminer conditions de marché
            is_near_level = min(distance_to_resistance, distance_to_support) < 0.005  # 0.5% de niveau clé
            is_high_volatility = volatility > 0.025  # >2.5% volatilité considérée élevée
            is_low_volatility = volatility < 0.01  # <1% volatilité considérée faible
            
            # Analyse tendance récente (5 dernières périodes)
            if len(data) >= 5:
                recent_prices = data['close'].iloc[-5:].values
                price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
                avg_change = sum(price_changes) / len(price_changes)
                trend_strength = abs(avg_change) / recent_prices[0]
                is_strong_trend = trend_strength > 0.002  # >0.2% mouvement moyen
            else:
                is_strong_trend = False
            
            # Adapter les poids selon les conditions
            if is_near_level:
                # Près d'un niveau clé: favoriser BREAKOUT et MEAN_REVERSION
                return {
                    'MOMENTUM': 0.20,
                    'BREAKOUT': 0.40,
                    'MEAN_REVERSION': 0.35,
                    'ORDER_BOOK': 0.05
                }
            elif is_high_volatility:
                # Haute volatilité: privilégier MOMENTUM et réduire MEAN_REVERSION
                return {
                    'MOMENTUM': 0.45,
                    'BREAKOUT': 0.35,
                    'MEAN_REVERSION': 0.10,
                    'ORDER_BOOK': 0.10
                }
            elif is_low_volatility:
                # Faible volatilité: privilégier MEAN_REVERSION et ORDER_BOOK
                return {
                    'MOMENTUM': 0.15,
                    'BREAKOUT': 0.20,
                    'MEAN_REVERSION': 0.50,
                    'ORDER_BOOK': 0.15
                }
            elif is_strong_trend:
                # Tendance forte: favoriser MOMENTUM
                return {
                    'MOMENTUM': 0.50,
                    'BREAKOUT': 0.30,
                    'MEAN_REVERSION': 0.15,
                    'ORDER_BOOK': 0.05
                }
            else:
                # Conditions équilibrées: répartition plus égale
                return {
                    'MOMENTUM': 0.30,
                    'BREAKOUT': 0.30,
                    'MEAN_REVERSION': 0.30,
                    'ORDER_BOOK': 0.10
                }
                
        except Exception as e:
            self.logger.error(f"Erreur calcul poids adaptatifs: {str(e)}")
            # En cas d'erreur, retourner des poids équilibrés
            return {
                'MOMENTUM': 0.25,
                'BREAKOUT': 0.25,
                'MEAN_REVERSION': 0.25,
                'ORDER_BOOK': 0.25
            }

    def _analyze_breakout(self, data: pd.DataFrame, tech_analysis: Dict) -> float:
        """Analyse les conditions de breakout."""
        try:
            score = 0.0
            
            # Indicateurs bollinger
            price_levels = tech_analysis.get('price_levels', {})
            if price_levels:
                resistance = price_levels.get('resistance', 0)
                support = price_levels.get('support', 0)
                current = float(data['close'].iloc[-1])

                # Distances relatives aux niveaux
                if resistance > 0:
                    distance_to_resistance = (resistance - current) / current
                    # Détection précoce de breakout haussier
                    if distance_to_resistance < 0.005:  # Très proche de la résistance
                        score += 0.4
                    # Breakout effectif haussier
                    if current > resistance:
                        score += 0.7
                
                if support > 0:
                    distance_to_support = (current - support) / current
                    # Détection précoce de breakout baissier
                    if distance_to_support < 0.005:  # Très proche du support
                        score -= 0.4
                    # Breakout effectif baissier
                    if current < support:
                        score -= 0.7
                
                # Confirmation volume
                volume_profile = tech_analysis.get('volume_profile', {})
                if volume_profile:
                    volume_ma = volume_profile.get('volume_ma', 0)
                    last_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
                    
                    # Volume confirmation (volume > moyenne)
                    if last_volume > volume_ma * 1.3:  # 30% au-dessus de la moyenne
                        score = score * 1.4 if score != 0 else 0.2  # Amplification ou score minimal
            
            # Utiliser la volatilité pour amplifier le signal
            volatility = tech_analysis.get('volatility', 0)
            if volatility > 0.01:  # Volatilité significative
                score *= 1.2  # Amplification modérée
                
            # Amplification finale
            return score * 3.0
                
        except Exception as e:
            self.logger.error(f"Erreur analyse breakout: {str(e)}")
            return 0.0

    def _analyze_mean_reversion(self, data: pd.DataFrame, tech_analysis: Dict) -> float:
        """Version optimisée de l'analyse mean_reversion."""
        try:
            score = 0.0
            
            # Utilisation des niveaux de prix pour mean reversion
            price_levels = tech_analysis.get('price_levels', {})
            if price_levels:
                current = float(data['close'].iloc[-1])
                support = price_levels.get('support', 0)
                resistance = price_levels.get('resistance', 0)
                
                # Si support et résistance sont définis, calculer la moyenne et la distance
                if support > 0 and resistance > 0:
                    mean_price = (support + resistance) / 2
                    distance = (current - mean_price) / mean_price
                    
                    # Score inversement proportionnel à la distance (tendance à revenir vers la moyenne)
                    score = -distance * 0.8
                    
                    # Amplifier si proche des extrêmes
                    if current > resistance * 0.99:  # Très proche/au-dessus de la résistance
                        score -= 0.5  # Plus forte probabilité de retour
                    elif current < support * 1.01:  # Très proche/en-dessous du support
                        score += 0.5  # Plus forte probabilité de retour
            
            # Utiliser la volatilité
            volatility = tech_analysis.get('volatility', 0)
            if volatility > 0.015:  # Volatilité élevée favorise mean-reversion
                score *= 1.5
                
            # Analyse des derniers mouvements (tendance à l'inversion après mouvement fort)
            if len(data) >= 5:
                # Calculer si les 3 dernières bougies sont dans la même direction
                last_prices = data['close'].iloc[-4:].values
                if all(last_prices[i] > last_prices[i-1] for i in range(1, len(last_prices))):
                    # 3 hausses consécutives, probable retournement à la baisse
                    score -= 0.4
                elif all(last_prices[i] < last_prices[i-1] for i in range(1, len(last_prices))):
                    # 3 baisses consécutives, probable retournement à la hausse
                    score += 0.4
            
            # Amplification finale
            return score * 3.0
                
        except Exception as e:
            self.logger.error(f"Erreur analyse mean reversion: {str(e)}")
            return 0.0

    def _analyze_momentum(self, data: pd.DataFrame, tech_analysis: Dict) -> float:
        """Analyse du momentum avec corrections critiques."""
        try:
            score = 0.0
            
            # Log de debug pour tracer l'exécution
            self.logger.info(f"Momentum: Début analyse avec {len(data)} points")
            
            # Tendance récente (5 périodes) - Simplification et robustesse
            if len(data) >= 5:
                # Calcul simplifié utilisant la direction des prix
                closes = data['close'].values
                direction = np.sign(closes[-1] - closes[-5])
                magnitude = abs(closes[-1] - closes[-5]) / closes[-5]
                
                # Score basé sur direction et magnitude
                score = direction * min(magnitude * 10, 1.0)
                
                self.logger.info(f"Momentum: Score calculé {score}")

            # Analyse du MACD pour confirmation de momentum
            macd_value = tech_analysis.get('macd', {}).get('macd', 0)
            macd_signal = tech_analysis.get('macd', {}).get('signal', 0)
            macd_hist = tech_analysis.get('macd', {}).get('hist', 0)
           
            if isinstance(macd_value, dict):
                # Si macd_value est un dict (format renvoyé par _calculate_indicators)
                macd_value = macd_value.get('macd', 0)
                macd_signal = macd_value.get('signal', 0)
                macd_hist = macd_value.get('hist', 0)
           
            # Utiliser le MACD pour renforcer le signal
            if macd_value > macd_signal and macd_hist > 0:
                score += 0.3 * min(abs(macd_hist / macd_value) * 10, 1.0) if macd_value != 0 else 0.3
            elif macd_value < macd_signal and macd_hist < 0:
                score -= 0.3 * min(abs(macd_hist / macd_value) * 10, 1.0) if macd_value != 0 else 0.3
           
            # Utiliser la force de tendance EMA
            ema_trend = tech_analysis.get('trend', 0)
            if abs(ema_trend) > self.config['trend']['min_trend_strength']:
                score += np.sign(ema_trend) * min(abs(ema_trend) * 3, 0.5)
               
            # Utiliser la volatilité pour ajuster l'amplitude
            volatility = tech_analysis.get('volatility', 0)
            if 0.005 < volatility < 0.025:  # Plage idéale pour momentum
                score *= 1.3
            elif volatility > 0.03:  # Trop volatil, réduit le score
                score *= 0.8
               
            # Amplification finale
            return score * 3.0
           
        except Exception as e:
            self.logger.error(f"Erreur analyse momentum: {str(e)}")
            return 0.001  # Valeur minimale non-nulle

    def _analyze_orderbook(self, tech_analysis: Dict) -> float:
        """Analyse order book optimisée pour la génération de signaux."""
        try:
            score = 0.0
           
            # Utiliser le profil de volume comme proxy pour le carnet d'ordres
            volume_profile = tech_analysis.get('volume_profile', {})
            if volume_profile:
                # Récupérer les métriques de volume
                volume_ma = volume_profile.get('volume_ma', 0)
                volume_std = volume_profile.get('volume_std', 0)
                volume_24h = volume_profile.get('volume_24h', 0)
               
                # Calculer ratio de volatilité du volume (indicateur de déséquilibre potentiel)
                if volume_ma > 0:
                    vol_ratio = volume_std / volume_ma
                   
                    # Score basé sur la volatilité du volume (corrélé aux déséquilibres)
                    if vol_ratio > 0.5:  # Forte variation = déséquilibre potentiel
                        score += vol_ratio * 0.8
                   
                # Volume actuel vs moyenne (indicateur de pression)
                last_volume = volume_24h / 24  # approximation horaire
                if volume_ma > 0 and last_volume > 0:
                    volume_pressure = last_volume / volume_ma
                   
                    # Score basé sur la pression de volume
                    if volume_pressure > 1.3:  # 30% au-dessus de la moyenne
                       score += 0.3
                    elif volume_pressure < 0.7:  # 30% en-dessous de la moyenne
                        score -= 0.3
           
            # Utiliser les niveaux de prix pour simuler des zones de demande/offre
            price_levels = tech_analysis.get('price_levels', {})
            if price_levels:
                # Récupérer les niveaux significatifs
                significant_levels = price_levels.get('significant_levels', [])
                current_price = price_levels.get('current_price', 0)
               
                if current_price > 0 and significant_levels:
                    # Calculer la densité des niveaux autour du prix actuel
                    nearby_levels = sum(1 for level in significant_levels 
                                        if abs(level - current_price) / current_price < 0.01)
                   
                    # Plus de niveaux à proximité = plus de liquidité potentielle
                    score += nearby_levels * 0.15
                   
            # Amplification finale du score
            return score * 3.0
               
        except Exception as e:
            self.logger.error(f"Erreur analyse orderbook: {str(e)}")
            return 0.0

    def _update_metrics(self, strategy_type: str = None):
        """Met à jour les métriques de génération de signaux."""
        self.metrics.generated += 1
       
        # Mise à jour des métriques par stratégie
        if strategy_type and strategy_type in self.metrics.strategy_signals:
            self.metrics.strategy_signals[strategy_type] += 1
           
        # Nettoyage périodique du cache
        now = datetime.now()
        if (now - self.last_reset) > timedelta(hours=1):
            # Reset des compteurs horaires
            self.last_reset = now
            # Réinitialisation des compteurs de signaux par heure
            self.signal_counts = {}

    def _cleanup_cache(self):
        """Nettoie le cache des indicateurs expirés."""
        try:
            now = datetime.now()
           
            # Vérifier si le nettoyage est nécessaire
            if (now - self.last_cache_cleanup) < timedelta(minutes=5):
                return
               
            # Mettre à jour timestamp
            self.last_cache_cleanup = now
           
            # Nettoyer les entrées expirées
            expired_keys = []
            for key, cache_entry in self.indicator_cache.items():
                if now - cache_entry.get('timestamp', now) > self.cache_expiry:
                    expired_keys.append(key)
                   
            # Supprimer les entrées expirées
            for key in expired_keys:
                del self.indicator_cache[key]
               
            self.logger.debug(f"Cache nettoyé: {len(expired_keys)} entrées supprimées")
               
        except Exception as e:
            self.logger.error(f"Erreur nettoyage cache: {str(e)}")

    def get_metrics(self) -> Dict:
        """Retourne les métriques de génération de signaux."""
        return {
            'generated': self.metrics.generated,
            'filtered': self.metrics.filtered,
            'executed': self.metrics.executed,
            'rejected': self.metrics.rejected,
            'errors': self.metrics.errors,
            'strategy_breakdown': self.metrics.strategy_signals
        }

    def reset_metrics(self):
        """Réinitialise les métriques."""
        self.metrics = SignalMetrics()
        self.last_reset = datetime.now()
        self.logger.info("Métriques réinitialisées")

    def get_strategy_performance(self) -> Dict[str, float]:
        """
        Calcule la performance relative de chaque stratégie.
        Utile pour ajuster dynamiquement les poids.
        """
        try:
            strategy_perf = {}
            total = sum(self.metrics.strategy_signals.values())
           
            if total > 0:
                for strategy, count in self.metrics.strategy_signals.items():
                    strategy_perf[strategy] = count / total
            else:
                # Valeurs par défaut si aucun signal
                strategy_perf = {
                    'MOMENTUM': 0.25,
                    'MEAN_REVERSION': 0.25,
                    'BREAKOUT': 0.25,
                    'ORDER_BOOK': 0.25
                }
               
            return strategy_perf
               
        except Exception as e:
            self.logger.error(f"Erreur calcul performance stratégies: {str(e)}")
            return {
                'MOMENTUM': 0.25,
                'MEAN_REVERSION': 0.25,
                'BREAKOUT': 0.25,
                'ORDER_BOOK': 0.25
            }

    def optimize_weights(self):
        """
        Optimise les poids des stratégies en fonction de leur performance.
        Ajuste les poids pour favoriser les stratégies les plus performantes.
        """
        try:
            # Obtenir performance actuelle
            perf = self.get_strategy_performance()
           
            # Ajuster les poids avec un facteur d'amplification
            adjusted_weights = {}
            for strategy, ratio in perf.items():
                # Amplifier les différences avec un facteur 1.5
                adjusted_weights[strategy] = ratio ** 1.5
               
            # Normaliser les poids
            total = sum(adjusted_weights.values())
            if total > 0:
                for strategy in adjusted_weights:
                    adjusted_weights[strategy] /= total
                   
                # Mettre à jour la configuration
                self.config['strategy_weights'] = adjusted_weights
                self.logger.info(f"Poids optimisés: {adjusted_weights}")
               
            return adjusted_weights
               
        except Exception as e:
            self.logger.error(f"Erreur optimisation poids: {str(e)}")
            return self.config['strategy_weights']