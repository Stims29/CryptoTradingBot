#!/usr/bin/env python
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import asyncio

from ..exceptions import OrderError
from ..utils.logging import DetailedLogger
from ..exchange.binance_client import BinanceClient
from ..config import config

class OrderManager:
    def __init__(self, simulation_mode: bool = True):
        self.logger = logging.getLogger(__name__)
        self.simulation_mode = simulation_mode

        # Configuration complète
        self.config = {
            'order_limits': {
                'min_size': 0.05,
                'max_size': 3.0,
                'min_notional': 10.0,
                'size_precision': 8,
                'price_precision': 8
            },
            'execution': {
                'timeout': 30,
                'retry_attempts': 3
            }
        }

        # Initialisation des paramètres depuis la config
        self.min_order_size = self.config['order_limits']['min_size']
        self.max_order_size = self.config['order_limits']['max_size']
        self.size_precision = self.config['order_limits']['size_precision']
        self.price_precision = self.config['order_limits']['price_precision']

        # État de l'ordre manager
        self.is_active = True
        self.orders = []
        self.active_orders = {}
        self.order_history = []
        self.order_count = 0
        self.filled_orders = 0
        self.rejected_orders = []

        # Compteurs
        self.order_count = 0
        self.filled_orders = 0
        self.reject_count = 0  # Renommé pour cohérence

        # Configuration des stratégies
        self.strategy_config = {
            'BREAKOUT': {
                'fill_delay': 0.1,
                'fill_probability': 0.95,
                'partial_probability': 0.2,
                'slippage_range': (-0.001, 0.001)
            },
            'MEAN_REVERSION': {
                'fill_delay': 0.2,
                'fill_probability': 0.90,
                'partial_probability': 0.3,
                'slippage_range': (-0.002, 0.002)
            },
            'MOMENTUM': {
                'fill_delay': 0.15,
                'fill_probability': 0.93,
                'partial_probability': 0.25,
                'slippage_range': (-0.0015, 0.0015)
            },
            'ORDER_BOOK': {
                'fill_delay': 0.05,
                'fill_probability': 0.98,
                'partial_probability': 0.15,
                'slippage_range': (-0.0005, 0.0005)
            }
        }

        # Liste des symboles autorisés
        self.allowed_symbols = [
        'BTC/USDT', 'ETH/USDT', 'XRP/USDT',
        'DOT/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT',
        'UNI/USDT', 'AAVE/USDT', 'LINK/USDT', 'SUSHI/USDT', 'CRV/USDT',
        'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'ENJUSDT',
        'ETH/BTC', 'ADA/BTC', 'XRP/BTC', 'DOT/BTC'
        ]
        
        # Configuration des ordres
        self.min_order_size = 0.05
        self.max_order_size = 3.0
        self.order_timeout = 30
        self.price_precision = 8
        self.size_precision = 8
    
        # État de l'ordre manager
        self.orders = []
        self.active_orders = {}  # Ajout de active_orders
        self.order_count = 0
        self.filled_orders = 0
        self.rejected_orders = 0

    async def place_order(self, symbol: str, side: str, size: float, strategy: str) -> Optional[Dict]:
        """Place un ordre."""
        try:
            if not all([symbol, side, size, strategy]):
                raise ValueError("Paramètres d'ordre incomplets")
            order = {
                'id': f"order_{self.order_count}",
                'symbol': symbol,
                'side': side,
                'size': size,
                'strategy': strategy,
                'status': 'FILLED' if self.simulation_mode else 'NEW',
                'timestamp': datetime.now().isoformat(),
                'fills': [],  # Ajout des fills
                'created_at': datetime.now(),  # Ajout timestamp création
                'updated_at': datetime.now()   # Ajout timestamp mise à jour
            }

            # Arrondir la taille à la précision correcte
            size = round(size, self.config['order_limits']['size_precision'])
            
            # Log détaillé avant vérification
            self.logger.debug(
                f"Tentative placement ordre:\n"
                f"Symbole: {symbol}\n"
                f"Side: {side}\n"
                f"Taille: {size}\n"
                f"Min: {self.config['order_limits']['min_size']}\n"
                f"Max: {self.config['order_limits']['max_size']}"
            )

            # Vérification de la taille
            if size < self.config['order_limits']['min_size']:
                adjusted_size = self.config['order_limits']['min_size']
                self.logger.info(f"Ajustement taille ordre au minimum: {adjusted_size}")
                size = adjusted_size
            elif size > self.config['order_limits']['max_size']:
                adjusted_size = self.config['order_limits']['max_size']
                self.logger.info(f"Ajustement taille ordre au maximum: {adjusted_size}")
                size = adjusted_size
            
            # Création de l'ordre
            order = {
                'id': f"order_{self.order_count}",
                'symbol': symbol,
                'side': side,
                'size': size,
                'strategy': strategy,
                'status': 'FILLED' if self.simulation_mode else 'NEW',
                'timestamp': datetime.now().isoformat()
            }

            # Ajout aux statistiques
            self.order_count += 1
            if order['status'] == 'FILLED':
                self.filled_orders += 1

            self.logger.info(
                f"Ordre placé avec succès:\n"
                f"ID: {order['id']}\n"
                f"Symbole: {symbol}\n"
                f"Side: {side}\n"
                f"Taille: {size}\n"
                f"Stratégie: {strategy}"
            )

            return order

        except Exception as e:
            self.rejected_orders += 1
            self.logger.error(f"Erreur création ordre: {str(e)}")
            raise

    async def _simulate_execution(self, order: Dict) -> None:
        """Simule l'exécution d'un ordre."""
        try:
            # Configuration stratégie
            config = self.strategy_config.get(
                order['strategy'],
                self.strategy_config['BREAKOUT']  # Default config
            )
            
            # Délai simulé
            await asyncio.sleep(config['fill_delay'])
            
            # Simulation fill
            if np.random.random() < config['fill_probability']:
                # Base price (toujours avoir un prix)
                if order['price'] is not None:
                    base_price = order['price']
                elif order['fills'] and order['fills'][-1].get('price'):
                    base_price = order['fills'][-1]['price']
                else:
                    base_price = 100.0  # Prix par défaut
                
                # Calcul slippage
                slippage_min, slippage_max = config['slippage_range']
                slippage = np.random.uniform(slippage_min, slippage_max)
                
                # Prix avec slippage
                fill_price = base_price * (1 + (slippage if order['side'] == 'buy' else -slippage))
                
                # Fill partiel ?
                if np.random.random() < config['partial_probability']:
                    fill_size = order['size'] * np.random.uniform(0.3, 0.7)
                    remaining = order['size'] - fill_size
                    
                    if remaining < self.min_order_size:  # Éviter petits restes
                        fill_size = order['size']
                        remaining = 0
                        
                    order['filled_size'] = fill_size
                    order['remaining_size'] = remaining
                    order['status'] = 'PARTIALLY_FILLED' if remaining > 0 else 'FILLED'
                else:
                    order['filled_size'] = order['size']
                    order['remaining_size'] = 0.0
                    order['status'] = 'FILLED'
                
                # Ajout du fill
                fill = {
                    'price': fill_price,
                    'size': order['filled_size'],
                    'timestamp': datetime.now()
                }
                order['fills'].append(fill)
                
                # Prix moyen
                total_value = sum(f['price'] * f['size'] for f in order['fills'])
                total_size = sum(f['size'] for f in order['fills'])
                order['fill_price'] = total_value / total_size if total_size > 0 else None
                
            else:
                order['status'] = 'REJECTED'
                self.reject_count += 1
                rejected_order = {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'strategy': order['strategy'],
                    'reason': 'simulation_reject',
                    'timestamp': datetime.now()
                }
                self.rejected_orders.append(rejected_order)
                self.logger.warning(f"Ordre rejeté: {rejected_order}")

        except Exception as e:
            self.logger.error(f"Erreur simulation ordre: {str(e)}")
            order['status'] = 'ERROR'
            order['error'] = str(e)

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre."""
        try:
            if not self.is_active:
                raise OrderError("OrderManager non actif")
                
            if order_id not in self.active_orders:
                raise OrderError(f"Ordre non trouvé: {order_id}")
                
            order = self.active_orders[order_id]
            
            # Mode réel
            if not self.simulation_mode:
                try:
                    result = self.binance.cancel_order(
                        symbol=order['symbol'],
                        order_id=order.get('binance_id')
                    )
                    if not result:
                        raise OrderError("Erreur annulation Binance")
                except Exception as e:
                    self.logger.error(f"Erreur Binance annulation: {e}")
                    return False
            
            order['status'] = 'CANCELED'
            order['updated_at'] = datetime.now()
            self.order_history.append(order)
            del self.active_orders[order_id]
                
            self.logger.info(f"Ordre annulé: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur annulation ordre: {str(e)}")
            return False

    def get_order(self, order_id: str) -> Dict:
        """Récupère les détails d'un ordre."""
        try:
            # Vérifier ordres actifs
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Mode réel - mise à jour depuis Binance
                if not self.simulation_mode and order.get('binance_id'):
                    try:
                        result = self.binance.get_order(
                            symbol=order['symbol'],
                            order_id=order['binance_id']
                        )
                        if result:
                            order.update({
                                'status': result['status'],
                                'filled_size': result['filled'],
                                'fill_price': result['price']
                            })
                    except Exception as e:
                        self.logger.error(f"Erreur Binance get_order: {e}")
                
                return order
                
            # Recherche dans historique
            for order in self.order_history:
                if order['id'] == order_id:
                    return order
                    
            raise OrderError(f"Ordre non trouvé: {order_id}")
            
        except Exception as e:
            self.logger.error(f"Erreur récupération ordre: {str(e)}")
            return {}

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[Dict]:
        """Récupère les ordres ouverts."""
        try:
            # Mode réel - mise à jour depuis Binance
            if not self.simulation_mode:
                try:
                    binance_orders = self.binance.get_open_orders(symbol)
                    for bo in binance_orders:
                        if bo['orderId'] in self.active_orders:
                            order = self.active_orders[bo['orderId']]
                            order.update({
                                'status': bo['status'],
                                'filled_size': float(bo['executedQty']),
                                'fill_price': float(bo['price']) if bo['price'] else None
                            })
                except Exception as e:
                    self.logger.error(f"Erreur Binance open orders: {e}")
            
            orders = list(self.active_orders.values())
            
            if symbol:
                orders = [o for o in orders if o['symbol'] == symbol]
                
            if strategy:
                orders = [o for o in orders if o['strategy'] == strategy]
                
            return orders
            
        except Exception as e:
            self.logger.error(f"Erreur récupération ordres ouverts: {str(e)}")
            return []

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """Récupère l'historique des ordres."""
        try:
            orders = self.order_history.copy()
            
            if symbol:
                orders = [o for o in orders if o['symbol'] == symbol]
                
            if strategy:
                orders = [o for o in orders if o['strategy'] == strategy]
                
            if status:
                orders = [o for o in orders if o['status'] == status]
                
            return orders
            
        except Exception as e:
            self.logger.error(f"Erreur récupération historique: {str(e)}")
            return []

    def get_fill_rate(self) -> float:
        """Calcule le taux de remplissage des ordres."""
        try:
            if self.order_count == 0:
                return 0.0
            return (self.filled_orders / self.order_count) * 100
        except Exception as e:
            self.logger.error(f"Erreur calcul fill rate: {e}")
            return 0.0

    def get_reject_rate(self) -> float:
        """Calcule le taux de rejet des ordres."""
        try:
            if self.order_count == 0:
                return 0.0
            return (self.reject_count / self.order_count) * 100
        except Exception as e:
            self.logger.error(f"Erreur calcul reject rate: {e}")
            return 0.0

    def get_metrics(self) -> Dict:
        """Retourne les métriques du gestionnaire."""
        try:
            return {
                'order_count': self.order_count,
                'fill_count': self.fill_count,
                'reject_count': self.reject_count,
                'fill_rate': self.get_fill_rate(),
                'reject_rate': self.get_reject_rate(),
                'active_orders': len(self.active_orders),
                'strategy_metrics': self._get_strategy_metrics()
            }
        except Exception as e:
            self.logger.error(f"Erreur récupération métriques: {str(e)}")
            return {}

    def _get_strategy_metrics(self) -> Dict:
        """Calcule les métriques par stratégie."""
        try:
            metrics = {}
            
            for strategy in self.strategy_config.keys():
                strategy_orders = [
                    o for o in self.order_history 
                    if o['strategy'] == strategy
                ]
                
                total = len(strategy_orders)
                if total == 0:
                    metrics[strategy] = {
                        'orders': 0,
                        'fill_rate': 0,
                        'avg_slippage': 0,
                        'avg_execution_time': timedelta(0)
                    }
                    continue
                
                fills = len([o for o in strategy_orders if o['status'] == 'FILLED'])
                
                # Calcul slippage moyen
                slippages = []
                execution_times = []
                
                for order in strategy_orders:
                    if order['status'] == 'FILLED' and order.get('fill_price'):
                        base_price = order['price'] or order['fills'][0]['price']
                        fill_price = order['fill_price']
                        
                        if order['side'] == 'buy':
                            slippage = (fill_price / base_price - 1) * 100
                        else:
                            slippage = (base_price / fill_price - 1) * 100
                            
                        slippages.append(slippage)
                        
                        execution_time = order['updated_at'] - order['created_at']
                        execution_times.append(execution_time)
                
                metrics[strategy] = {
                    'orders': total,
                    'fill_rate': (fills / total) * 100,
                    'avg_slippage': float(np.mean(slippages)) if slippages else 0,
                    'avg_execution_time': (
                        sum(execution_times, timedelta(0)) / len(execution_times)
                        if execution_times else timedelta(0)
                    )
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur calcul métriques stratégie: {str(e)}")
            return {}

    async def bulk_cancel(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> int:
        """Annule plusieurs ordres selon critères."""
        try:
            orders_to_cancel = self.get_open_orders(symbol, strategy)
            cancelled = 0
            
            for order in orders_to_cancel:
                if await self.cancel_order(order['id']):
                    cancelled += 1
                    
            return cancelled
            
        except Exception as e:
            self.logger.error(f"Erreur annulation multiple: {str(e)}")
            return 0

    async def close(self):
        """Ferme proprement le gestionnaire."""
        try:
            self.is_active = False
            
            # Annulation ordres actifs
            await self.bulk_cancel()
            
            # Déconnexion Binance si mode réel
            if not self.simulation_mode:
                try:
                    self.binance.close()
                except Exception as e:
                    self.logger.error(f"Erreur fermeture Binance: {e}")
            
            self.logger.info(
                f"OrderManager fermé proprement\n"
                f"Ordres totaux: {self.order_count}\n"
                f"Taux remplissage: {self.get_fill_rate():.1f}%\n"
                f"Taux rejet: {self.get_reject_rate():.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Erreur fermeture OrderManager: {str(e)}")