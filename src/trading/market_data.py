#!/usr/bin/env python
# src/trading/market_data.py

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

class MarketDataManager:
    def __init__(self, simulation_mode: bool = True):
        """Initialise le gestionnaire de données de marché."""
        self.simulation_mode = simulation_mode
        self.logger = logging.getLogger(__name__)

        # Initialisation du cache
        self.data_cache = {}
        self.cache_config = {
            'ttl': 60,  # 60 secondes
            'max_size': 1000,
            'cleanup_interval': 300  # 5 minutes
        }
        
        # Configuration de base des paires
        self.config = {
            'update_frequency': 1  # secondes
        }

        # Configuration des paires de trading
        self.pairs_config = {
            'MAJOR': {
                'BTC/USDT': {'base_price': 52000, 'volume': 100000},
            #    'ETH/USDT': {'base_price': 3200, 'volume': 50000},
            #    'XRP/USDT': {'base_price': 0.55, 'volume': 200000000}
            },
            #'ALTCOINS': {
            #    'DOT/USDT': {'base_price': 7.2, 'volume': 80000000},
            #    'ADA/USDT': {'base_price': 0.5, 'volume': 100000000},
            #    'SOL/USDT': {'base_price': 110, 'volume': 150000000},
            #    'AVAX/USDT': {'base_price': 35, 'volume': 70000000}
            #}
            #'DEFI': {
            #    'UNI/USDT': {'base_price': 7.5, 'volume': 50000000},
            #    'AAVE/USDT': {'base_price': 90, 'volume': 40000000},
            #   'LINK/USDT': {'base_price': 18, 'volume': 60000000},
            #    'SUSHI/USDT': {'base_price': 1.5, 'volume': 30000000},
            #    'CRV/USDT': {'base_price': 0.6, 'volume': 25000000}
            #},
            #'NFT_METAVERSE': {
            #    'SAND/USDT': {'base_price': 0.5, 'volume': 20000000},
            #    'MANA/USDT': {'base_price': 0.48, 'volume': 25000000},
            #    'AXS/USDT': {'base_price': 8.5, 'volume': 15000000},
            #    'ENJ/USDT': {'base_price': 0.35, 'volume': 10000000}
            #},
            #'BTC_PAIRS': {
            #    'ETH/BTC': {'base_price': 0.06, 'volume': 100000},
            #    'ADA/BTC': {'base_price': 0.00002, 'volume': 30000000},
            #    'XRP/BTC': {'base_price': 0.00003, 'volume': 20000000},
            #    'DOT/BTC': {'base_price': 0.00004, 'volume': 15000000}
            #}
        }

        # Configuration de la simulation
        self.sim_config = {
            'volatility': {
                'MAJOR': {'base': 0.0002, 'spike': 0.001},  # Triplé
                'ALTCOINS': {'base': 0.0004, 'spike': 0.004},  # Doublé
                'DEFI': {'base': 0.006, 'spike': 0.03},  # +66%
                'NFT_METAVERSE': {'base': 0.008, 'spike': 0.04},  # +50%
                'BTC_PAIRS': {'base': 0.003, 'spike': 0.015}  # Doublé
            },
            'volume_variation': {
                'base': 0.2,  # 20% variation de base
                'spike': 0.5  # 50% pics de volume
            },
            'trend_strength': {
                'weak': 0.0001,
                'medium': 0.0003,
                'strong': 0.0005
            },
            'update_frequency': 1  # secondes
        }

        self.logger.info(
            f"MarketDataManager initialisé\n"
            f"Mode simulation: {self.simulation_mode}\n"
            f"Nombre de paires: {sum(len(pairs) for pairs in self.pairs_config.values())}"
        )

    async def get_market_data(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Récupère les données de marché avec cache."""
        try:
            self.logger.info(f"Début récupération données {symbol}")
        
            cache_key = f"{symbol}_{interval}_{limit}"
        
            # Vérification cache
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if not self._is_cache_expired(cached_data['timestamp']):
                    return cached_data['data']
        
            # En mode simulation, générer les données
            if self.simulation_mode:
                data = self._generate_simulated_data(symbol, interval, limit)
            else:
                # En mode réel, récupérer les données
                data = await self._fetch_market_data(symbol, interval, limit)
        
            # Mise en cache
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }

            data = self._generate_simulated_data(symbol, interval, limit)
            self.logger.info(f"Données générées pour {symbol}: empty={data.empty}")
            if not data.empty:
                self.logger.info(f"Prix actuel: {data['close'].iloc[-1]}, Vol: {data['volume'].iloc[-1]}")
        
            return data
        
        except Exception as e:
            self.logger.error(f"Erreur récupération données {symbol}: {str(e)}")
            return pd.DataFrame()

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère les données du cache."""
        if key in self.data_cache:
            cached_time, data = self.data_cache[key]
            if datetime.now() - cached_time < self.cache_duration:
                return data
        return None

    def _cache_data(self, key: str, data: pd.DataFrame) -> None:
        """Met en cache les données de marché."""
        self.data_cache[key] = (datetime.now(), data)
        
        # Nettoyage périodique du cache
        if datetime.now() - self._last_cache_cleanup > timedelta(minutes=5):
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Nettoie les entrées expirées du cache."""
        now = datetime.now()
        expired_keys = [
            key for key, value in self.data_cache.items()
            if (now - value['timestamp']).seconds > self.cache_config['ttl']
        ]
        for key in expired_keys:
            del self.data_cache[key]

    def _is_cache_expired(self, timestamp: datetime) -> bool:
        """Vérifie si une entrée du cache est expirée."""
        return (datetime.now() - timestamp).seconds > self.cache_config['ttl']

    def _generate_simulated_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Génère des données simulées avec volatilité optimisée pour scalping."""
        self.logger.info(f"Génération données {symbol}")
       
        volatility = {
        'base': 0.005,  # 0.2% de volatilité base
        'spike': 0.01  # 0.5% pour les spikes
        }
        try:
            # Récupération configuration
            category = self._get_symbol_category(symbol)
            pair_config = self._get_pair_config(symbol)
            if not pair_config:
                return pd.DataFrame()
            
            base_price = pair_config['base_price']
            base_volume = pair_config['volume']
        
            # Paramètres de volatilité augmentés
            volatility = {
                'base': self.sim_config['volatility'][category]['base'] * 3,  # Triplé
                'spike': self.sim_config['volatility'][category]['spike'] * 2  # Doublé
            }
        
            # Génération des timestamps
            end_time = datetime.now()
            timestamps = [
                end_time - timedelta(minutes=i)
                for i in range(limit-1, -1, -1)
            ]
        
            # Génération prix
            prices = []
            volumes = []
            current_price = base_price
        
            # Création d'une mini-tendance directionnelle
            trend_direction = np.random.choice([-1, 1])
            trend_strength = np.random.uniform(0.0005, 0.002)  # 0.05%-0.2%
            trend_duration = np.random.randint(5, 15)  # Durée aléatoire de la tendance
            trend_active = False
            trend_count = 0
        
            for i in range(limit):
                # Gestion des tendances
                if i % trend_duration == 0:
                    # Nouvelle tendance
                    trend_direction = np.random.choice([-1, 1])
                    trend_strength = np.random.uniform(0.0005, 0.003)  # Augmenté
                    trend_duration = np.random.randint(5, 20)  # Plus variable
                    trend_active = np.random.random() < 0.5  # 50% chance de tendance active
                    trend_count = 0
                
                # Calcul du changement de prix de base
                price_change = 2.0 * np.random.normal(0, volatility['base'])  # facteur x2
            
                # Ajout de la tendance si active
                if trend_active and trend_count < trend_duration:
                    price_change += trend_direction * trend_strength
                    trend_count += 1
            
                # Pics de volatilité (10% de probabilité au lieu de 5%)
                if np.random.random() < 0.10:
                    price_change += 2.0 * np.random.normal(0, volatility['spike'])
                
                # Régimes de marché (bull/bear/crab)
                market_regime = np.random.choice(['bull', 'bear', 'crab'], p=[0.4, 0.3, 0.3])
                if market_regime == 'bull':
                    price_change += abs(price_change) * 0.2  # Biais haussier
                elif market_regime == 'bear':
                    price_change -= abs(price_change) * 0.2  # Biais baissier
                
                # Application changement de prix
                current_price *= (1 + price_change)
                prices.append(current_price)
            
                # Génération volume avec plus de variabilité
                volume_variation = np.random.normal(
                    0,
                    self.sim_config['volume_variation']['base'] * 1.5  # +50% de variation
                )
            
                # Pics de volume (15% de probabilité)
                if np.random.random() < 0.15:
                    volume_variation += np.random.uniform(0.5, 1.5) * self.sim_config['volume_variation']['spike']
                
                # Corrélation volume/mouvement de prix (volumes plus élevés avec grands mouvements)
                if abs(price_change) > volatility['base'] * 2:
                    volume_variation += abs(price_change) / volatility['base'] * 0.5
                
                volume = base_volume * (1 + volume_variation)
                volumes.append(max(volume, base_volume * 0.5))  # Garder un minimum
            
            # Création DataFrame avec OHLC réaliste
            highs = []
            lows = []
            opens = []
        
            # Premier open = premier prix
            opens.append(prices[0])
        
            # Génération OHLC réaliste
            for i in range(limit):
                # Calcul high et low en fonction de la volatilité
                price = prices[i]
                price_vol = price * volatility['base'] * np.random.uniform(2, 5)
            
                # High toujours supérieur au prix de clôture
                high = price + abs(price_vol) * np.random.uniform(0.5, 1.5)
            
                # Low toujours inférieur au prix de clôture
                low = price - abs(price_vol) * np.random.uniform(0.5, 1.5)
            
                highs.append(high)
                lows.append(low)
            
                # Génération de l'open pour la barre suivante (sauf dernière)
                if i < limit - 1:
                    # Open proche du close précédent
                    opens.append(price * (1 + np.random.normal(0, volatility['base'] * 0.3)))
        
            # Création DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes
            })
        
            return df
        
        except Exception as e:
            self.logger.error(f"Erreur génération données simulées {symbol}: {e}")
        return pd.DataFrame()

    async def _fetch_market_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Récupère les données réelles du marché."""
        # À implémenter avec l'API de l'exchange
        pass

    def calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calcule la volatilité sur les données."""
        try:
            if data.empty:
                return 0.0
                
            returns = np.log(data['close'] / data['close'].shift(1))
            return returns.std() * np.sqrt(len(data))
            
        except Exception as e:
            self.logger.error(f"Erreur calcul volatilité: {e}")
            return 0.0

    def calculate_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Calcule le profil de volume."""
        try:
            if data.empty:
                return {}
                
            return {
                'volume_24h': data['volume'].sum(),
                'volume_ma': data['volume'].rolling(window=20).mean().iloc[-1],
                'volume_std': data['volume'].rolling(window=20).std().iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul profil volume: {e}")
            return {}

    def calculate_price_levels(self, data: pd.DataFrame) -> Dict:
        """Calcule les niveaux de prix importants de manière optimisée."""
        try:
            if data.empty:
                return {}
            
            # Calcul niveaux - valeurs de base
            high = data['high'].max()
            low = data['low'].min()
            current = data['close'].iloc[-1]
        
            # Approche vectorisée pour l'identification des niveaux
            price_levels = pd.concat([data['high'], data['low']])
            hist, bins = np.histogram(price_levels, bins=20)  # Réduit de 50 à 20 bins
        
            # Calcul statistique fait une seule fois
            threshold = np.mean(hist) + np.std(hist)
        
            # Vectorisation complète - évite les boucles
            significant_indices = np.where(hist > threshold)[0]
            significant_levels = [(bins[i] + bins[i+1])/2 for i in significant_indices]
        
            # Support et résistance avec gestion des cas limites
            levels_below = [level for level in significant_levels if level < current]
            levels_above = [level for level in significant_levels if level > current]
        
            support = max(levels_below) if levels_below else low
            resistance = min(levels_above) if levels_above else high
        
            return {
                'support': float(support),
                'resistance': float(resistance),
                'significant_levels': significant_levels,
                'current_price': float(current)
            }
        
        except Exception as e:
            self.logger.error(f"Erreur calcul niveaux prix: {str(e)}")
            # Valeurs par défaut sécurisées
            return {
                'support': float(data['low'].min()) if not data.empty else 0.0,
                'resistance': float(data['high'].max()) if not data.empty else 0.0,
                'significant_levels': [],
                'current_price': float(data['close'].iloc[-1]) if not data.empty else 0.0
            }
    
    def _get_symbol_category(self, symbol: str) -> str:
        """Détermine la catégorie d'un symbole."""
        for category, pairs in self.pairs_config.items():
            if symbol in pairs:
                return category
        return 'ALTCOINS'  # Catégorie par défaut

    def _get_pair_config(self, symbol: str) -> Optional[Dict]:
        """Récupère la configuration d'une paire."""
        for pairs in self.pairs_config.values():
            if symbol in pairs:
                return pairs[symbol]
        return None

    def get_all_symbols(self) -> List[str]:
        """Retourne la liste de tous les symboles disponibles."""
        symbols = []
        for pairs in self.pairs_config.values():
            symbols.extend(pairs.keys())
        return symbols
    
    def get_all_symbols(self) -> List[str]:
        """Retourne la liste de tous les symboles disponibles."""
        symbols = []
        for pairs in self.pairs_config.values():
            symbols.extend(pairs.keys())
        return symbols

    async def monitor_cache_performance(self) -> Dict:
        """
        Monitore les performances du cache de données.
        Retourne les métriques de performance du cache.
        """
        try:
            hit_count = 0
            miss_count = 0
            total_latency = 0
            
            for symbol in self.get_all_symbols():
                start_time = time.time()
                data = await self.get_market_data(symbol)
                latency = time.time() - start_time
                
                if symbol in self.data_cache:
                    hit_count += 1
                else:
                    miss_count += 1
                total_latency += latency
            
            # Calcul métriques
            total_requests = hit_count + miss_count
            hit_rate = (hit_count / total_requests * 100) if total_requests > 0 else 0
            avg_latency = (total_latency / total_requests) if total_requests > 0 else 0
            
            metrics = {
                'hit_rate': hit_rate,
                'miss_rate': 100 - hit_rate,
                'avg_latency_ms': avg_latency * 1000,
                'cache_size': len(self.data_cache),
                'total_requests': total_requests
            }
            
            self.logger.info(
                f"Cache Performance:\n"
                f"Hit Rate: {hit_rate:.1f}%\n"
                f"Miss Rate: {100-hit_rate:.1f}%\n"
                f"Avg Latency: {avg_latency*1000:.2f}ms\n"
                f"Cache Size: {len(self.data_cache)} entries"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur monitoring cache: {str(e)}")
            return {}