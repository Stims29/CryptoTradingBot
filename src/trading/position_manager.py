#!/usr/bin/env python
# src/trading/position_manager.py

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

class PositionManager:
    def __init__(self, simulation_mode: bool = True, initial_capital: float = 10.0):
        """Initialise le gestionnaire de positions."""
        self.logger = logging.getLogger(__name__)
        self.simulation_mode = simulation_mode
        self.initial_capital = initial_capital
        
        # Initialisation des positions
        self.positions = {}
        
        # Initialisation des métriques
        self.metrics = {
            'total_pnl': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'active_positions': 0,
            'closed_positions': 0,
            'opened_positions': 0,
            'win_rate': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'avg_position_duration': 0.0,
            'sharpe_ratio': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        # Configuration par type de marché
        self.market_configs = {
            'MAJOR': {
                'stop_loss': 0.002,      # 0.2%
                'take_profit': 0.003,    # 0.3%
                'trailing_stop': 0.0005, # 0.05%
                'position_max': 0.01     # 1% du capital seulement
            },
            'ALTCOINS': {
                'stop_loss': 0.006,    # Augmenter de 0.0025 à 0.006 (0.6%)
                'take_profit': 0.012,   # Augmenter de 0.004 à 0.012 (1.2%)
                'trailing_stop': 0.002, # Réduit de 0.003/0.004 à 0.002 (0.2%)
                'position_max': 0.015    # Réduit de 0.02 à 0.015 (1.5% du capital)
            },
            'DEFI': {
                'stop_loss': 0.005,     # Réduit de 0.006/0.008 à 0.005 (0.5%)
                'take_profit': 0.012,   # Réduit de 0.015 à 0.012 (1.2%)
                'trailing_stop': 0.003, # Réduit de 0.004/0.006 à 0.003 (0.3%)
                'position_max': 0.01   # Réduit de 0.015 à 0.01 (1% du capital)
            },
            'NFT_METAVERSE': {
                'stop_loss': 0.006,    # Réduit de 0.007/0.010 à 0.006 (0.6%)
                'take_profit': 0.015,   # Réduit de 0.020 à 0.015 (1.5%)
                'trailing_stop': 0.004, # Réduit de 0.005/0.007 à 0.004 (0.4%)
                'position_max': 0.01   # Réduit de 0.015 à 0.01 (1% du capital)
            },
            'BTC_PAIRS': {
                'stop_loss': 0.003,     # Réduit de 0.004/0.005 à 0.003 (0.3%)
                'take_profit': 0.008,   # Réduit de 0.010 à 0.008 (0.8%)
                'trailing_stop': 0.0015, # Réduit de 0.002/0.003 à 0.0015 (0.15%)
                'position_max': 0.01   # Réduit de 0.015 à 0.01 (1% du capital)
            }
        }
        
        # Limites globales de risque
        self.risk_limits = {
            'max_drawdown': 0.25,         # Augmenté de 0.20 à 0.25
            'max_daily_loss': 0.20,       # Augmenté de 0.15 à 0.20
            'max_exposure': 0.15,         # Réduit de 0.25 à 0.15 (15% exposition maximale)
            'max_positions': 3,           # Réduit de 5 à 3 positions simultanées max
            'position_timeout': 30,      # Réduit à 30 sec. pour test
            'emergency_close': False,     # # Renommé de emergency_close à emergency_mode
            'recovery_threshold': 0.10    # Seuil de récupération pour sortie du mode urgence
        }
        
        # Statistiques des positions
        self.position_stats = {
            'win_count': 0,
            'loss_count': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'total_duration': 0,
            'position_count': 0
        }
        
        # Historique des positions pour analyse
        self.position_history = []
        
        self.logger.info(f"PositionManager initialisé en mode {'simulation' if simulation_mode else 'réel'} avec capital {initial_capital}€")

    async def open_position(self, symbol, side, size, entry_price, market_type):
        """
        Ouvre une nouvelle position de trading.
    
        Args:
            symbol (str): Symbole du marché (ex: BTC/USDT)
            side (str): Direction de la position ('buy' ou 'sell')
            size (float): Taille de la position en unités
            entry_price (float): Prix d'entrée
            market_type (str): Catégorie de marché (MAJOR, ALTCOINS, etc.)
        
        Returns:
            bool: True si la position a été ouverte avec succès, False sinon
        """
        try:
            self.logger.info(f"[DIAGNOSTIC] Demande ouverture position: {symbol}, {side}, {size}, {entry_price}, {market_type}")
            # Vérification des paramètres
            if not all([symbol, side, size, entry_price, market_type]):
                self.logger.error(f"Paramètres invalides pour open_position: {symbol}, {side}, {size}, {entry_price}, {market_type}")
                return False
            
            # Vérification si position existante
            if symbol in self.positions:
                # Au lieu de rejeter, vérifier si on peut ajouter à la position
                existing_pos = self.positions[symbol]
            
                # Si même direction, envisager d'augmenter la position
                if existing_pos['side'] == side:
                    # Limiter l'augmentation de position
                    max_position_value = self.initial_capital * self.market_configs.get(market_type, {}).get('position_max', 0.05)
                    current_position_value = existing_pos['size'] * entry_price
                
                    if current_position_value < max_position_value * 0.7:  # 70% de la taille max
                        # Possible d'ajouter à la position existante
                        self.logger.info(f"Augmentation position existante {symbol} de {existing_pos['size']} à {existing_pos['size'] + size}")
                    
                        # Mise à jour de la position
                        old_size = existing_pos['size']
                        new_size = old_size + size
                    
                        # Calcul du prix d'entrée moyen pondéré
                        existing_pos['avg_entry_price'] = (existing_pos['entry_price'] * old_size + entry_price * size) / new_size
                        existing_pos['size'] = new_size
                        existing_pos['last_update'] = datetime.now()
                    
                        # Recalcul des niveaux de sortie
                        self._update_exit_levels(existing_pos, market_type)
                    
                        return True
            
                self.logger.warning(f"Position déjà existante pour {symbol}, augmentation non possible")
                return False
                
            # Validation de la taille et du prix
            if size <= 0 or entry_price <= 0:
                self.logger.error(f"Taille ou prix invalide: {size}, {entry_price}")
                return False
            
            # Calcul des stop loss et take profit selon le type de marché
            sl_pct = self.market_configs.get(market_type, {}).get('stop_loss', 0.003)
            tp_pct = self.market_configs.get(market_type, {}).get('take_profit', 0.008)
        
            # Calcul des niveaux de sortie
            if side == 'buy':
                stop_loss = entry_price * (1 - sl_pct)
                take_profit = entry_price * (1 + tp_pct)
            else:  # 'sell'
                stop_loss = entry_price * (1 + sl_pct)
                take_profit = entry_price * (1 - tp_pct)
            
            # Création de la position
            position = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'original_size': size,  # Pour suivi des augmentations
                'entry_price': entry_price,
                'avg_entry_price': entry_price,  # Pour le calcul du prix moyen
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'market_type': market_type,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'status': 'open',
                'entry_time': datetime.now(),
                'last_update': datetime.now(),
                'duration': 0,
                'max_price': entry_price,
                'min_price': entry_price,
                'partial_exits': [],  # Pour tracking des sorties partielles
                'increased': False    # Indique si la position a été augmentée
            }
        
            # Enregistrement de la position
            self.positions[symbol] = position
        
            # Mise à jour des métriques
            self.metrics['active_positions'] = len(self.positions)
            self.metrics['opened_positions'] += 1
        
            # Log détaillé
            self.logger.info(f"Position ouverte: {symbol} {side} à {entry_price}, taille={size}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
        
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur ouverture position {symbol}: {str(e)}")
            return False

    def _update_exit_levels(self, position, market_type):
        """Met à jour les niveaux de sortie d'une position."""
        try:
            sl_pct = self.market_configs.get(market_type, {}).get('stop_loss', 0.003)
            tp_pct = self.market_configs.get(market_type, {}).get('take_profit', 0.008)
        
            # Recalcul basé sur le prix d'entrée moyen
            if position['side'] == 'buy':
                position['stop_loss'] = position['avg_entry_price'] * (1 - sl_pct)
                position['take_profit'] = position['avg_entry_price'] * (1 + tp_pct)
            else:  # 'sell'
                position['stop_loss'] = position['avg_entry_price'] * (1 + sl_pct)
                position['take_profit'] = position['avg_entry_price'] * (1 - tp_pct)
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour niveaux sortie: {str(e)}")

    async def update_position(self, position_data: Dict) -> float:
        """
        Met à jour une position existante avec le prix actuel.
    
        Args:
            position_data (Dict): Données de mise à jour avec au moins 'symbol' et 'current_price'
            
        Returns:
            float: PnL réalisé si position fermée, sinon 0
        """
        try:
            symbol = position_data.get('symbol')
            current_price = position_data.get('current_price')
        
            if not symbol or not current_price:
                self.logger.error(f"Données mise à jour position incomplètes: {position_data}")
                return 0.0
                
            if symbol not in self.positions:
                self.logger.warning(f"Position inexistante pour mise à jour: {symbol}")
                return 0.0
                
            position = self.positions[symbol]
        
            # Mise à jour du prix actuel
            position['current_price'] = current_price
            position['last_update'] = datetime.now()
        
            # Mise à jour des prix min/max
            position['max_price'] = max(position['max_price'], current_price)
            position['min_price'] = min(position['min_price'], current_price)
        
            # Calcul de la durée
            duration = (position['last_update'] - position['entry_time']).total_seconds()
            position['duration'] = duration
        
            # Calcul du PnL non réalisé
            if position['side'] == 'buy':
                unrealized_pnl = (current_price - position['entry_price']) * position['size']
            else:  # 'sell'
                unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                
            position['unrealized_pnl'] = unrealized_pnl
        
            # Stop loss d'urgence basé sur le pourcentage du capital initial
            if hasattr(self, 'initial_capital') and self.initial_capital > 0:
                if unrealized_pnl < -0.01 * self.initial_capital:
                    self.logger.warning(f"Stop loss d'urgence activé pour {symbol}: Perte > 1% du capital initial")
                    return await self.close_position(symbol, current_price, "emergency_stop")
        
            # Ajout du trailing stop
            try:
                market_type = position['market_type']
                trailing_stop_pct = self.market_configs.get(market_type, {}).get('trailing_stop', 0.001)
            
                # Mise à jour du trailing stop pour position acheteuse
                if position['side'] == 'buy' and current_price > position['entry_price']:
                    # Calculer le nouveau stop basé sur le prix max
                    new_stop = position['max_price'] * (1 - trailing_stop_pct)
                    # Ne mettre à jour que si le nouveau stop est supérieur à l'ancien
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        self.logger.info(f"Trailing stop mis à jour: {symbol} à {new_stop:.2f}")
            
                # Mise à jour du trailing stop pour position vendeuse
                elif position['side'] == 'sell' and current_price < position['entry_price']:
                    # Calculer le nouveau stop basé sur le prix min
                    new_stop = position['min_price'] * (1 + trailing_stop_pct)
                    # Ne mettre à jour que si le nouveau stop est inférieur à l'ancien
                    if new_stop < position['stop_loss']:
                        position['stop_loss'] = new_stop
                        self.logger.info(f"Trailing stop mis à jour: {symbol} à {new_stop:.2f}")
            except Exception as e:
                self.logger.error(f"Erreur calcul trailing stop pour {symbol}: {str(e)}")
        
            # Vérification du timeout
            try:
                # Vérifier si la position a dépassé sa durée maximale
                if hasattr(self, 'risk_limits') and 'position_timeout' in self.risk_limits:
                    max_duration = self.risk_limits['position_timeout']
                    if duration > max_duration:
                        self.logger.warning(f"Position {symbol} a atteint timeout ({duration:.1f}s > {max_duration}s)")
                        return await self.close_position(symbol, current_price, "timeout")
            except Exception as e:
                self.logger.error(f"Erreur vérification timeout pour {symbol}: {str(e)}")
                    
            # Vérification stop loss et take profit
            triggered = None
            if position['side'] == 'buy':
                if current_price <= position['stop_loss']:
                    triggered = 'stop_loss'
                elif current_price >= position['take_profit']:
                    triggered = 'take_profit'
            else:  # 'sell'
                if current_price >= position['stop_loss']:
                    triggered = 'stop_loss'
                elif current_price <= position['take_profit']:
                    triggered = 'take_profit'
                    
            # Fermeture si stop ou take profit atteint
            if triggered:
                self.logger.info(f"Position {symbol} {triggered} atteint. Fermeture auto à {current_price}")
                return await self.close_position(symbol, current_price, triggered)
                
            # Mise à jour des métriques globales
            self._update_metrics()
        
            return 0.0
                
        except Exception as e:
            self.logger.error(f"Erreur mise à jour position {position_data.get('symbol', 'unknown')}: {str(e)}")
            return 0.0

    async def close_position(self, symbol: str, exit_price: float, reason: str = 'manual') -> float:
        """
        Ferme une position existante.
    
        Args:
            symbol (str): Symbole de la position à fermer
            exit_price (float): Prix de sortie
            reason (str): Raison de la fermeture
        
        Returns:
            float: PnL réalisé
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"Position inexistante pour fermeture: {symbol}")
                return 0.0
                
            position = self.positions[symbol]
        
            # Calculer le PnL réalisé
            if position['side'] == 'buy':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:  # 'sell'
                pnl = (position['entry_price'] - exit_price) * position['size']

            self.logger.info(f"[DIAGNOSTIC] PnL réalisé: {symbol} - {pnl:.6f}€ [avant mise à jour des métriques]")
            self.logger.info(f"[DIAGNOSTIC] AVANT fermeture - metrics['total_pnl']: {self.metrics['total_pnl']:.6f}€")

            # Traçage détaillé des positions fermées
            self.logger.info(f"FERMETURE DÉTAILLÉE: {symbol} - Entrée: {position['entry_price']}, Sortie: {exit_price}, Côté: {position['side']}, Taille: {position['size']}, PnL: {pnl:.6f}€, Raison: {reason}")
            
            # Log détaillé
            self.logger.info(f"[TEST 3D] Fermeture position {symbol} - Prix entrée: {position['entry_price']}, Prix sortie: {exit_price}, PnL calculé: {pnl:.6f}")
        
            # Avant de retourner le PnL
            self.logger.info(f"[TEST 3D] PnL réalisé retourné: {pnl:.6f}")
        
            # Protection contre les PnL irréalistes
            if hasattr(self, 'initial_capital') and self.initial_capital > 0:
                if abs(pnl) > self.initial_capital * 0.5:  # Si le PnL dépasse 50% du capital initial
                    self.logger.warning(f"PnL anormal détecté pour {symbol}: {pnl:.2f}€. Limité à 10% du capital initial.")
                    pnl = self.initial_capital * 0.1 * (1 if pnl > 0 else -1)  # Limiter à ±10% du capital
            
            # Mise à jour position
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['realized_pnl'] = pnl
            position['reason'] = reason
        
            # Calcul durée finale
            duration = (position['exit_time'] - position['entry_time']).total_seconds()
            position['duration'] = duration
        
            # Récupération de la stratégie utilisée
            strategy = position.get('strategy')
        
            # Mise à jour des performances de la stratégie si possible
            if strategy and hasattr(self, 'bot') and hasattr(self.bot, 'strategy'):
                is_win = pnl > 0
                self.bot.strategy.update_strategy_performance(strategy, is_win)
        
            # Mettre à jour statistiques
            if pnl > 0:
                self.position_stats['win_count'] += 1
                self.position_stats['avg_win'] = ((self.position_stats['avg_win'] * (self.position_stats['win_count'] - 1)) + pnl) / self.position_stats['win_count'] if self.position_stats['win_count'] > 0 else pnl
                self.position_stats['max_win'] = max(self.position_stats['max_win'], pnl)

                # Mettre à jour le meilleur trade
                self.metrics['best_trade'] = max(self.metrics.get('best_trade', 0), pnl)
            else:
                self.position_stats['loss_count'] += 1
                self.position_stats['avg_loss'] = ((self.position_stats['avg_loss'] * (self.position_stats['loss_count'] - 1)) + pnl) / self.position_stats['loss_count'] if self.position_stats['loss_count'] > 0 else pnl
                self.position_stats['max_loss'] = min(self.position_stats['max_loss'], pnl)

                # Mettre à jour le pire trade
                self.metrics['worst_trade'] = min(self.metrics.get('worst_trade', 0), pnl)
            
            self.position_stats['total_duration'] += duration
            self.position_stats['position_count'] += 1
        
            # Mettre à jour métriques
            self.metrics['realized_pnl'] += pnl
            self.metrics['total_pnl'] += pnl

            # Après l'update des métriques
            self.logger.info(f"[DIAGNOSTIC] APRÈS fermeture - metrics['total_pnl']: {self.metrics['total_pnl']:.6f}€")
        
            # Ajouter à l'historique
            self.position_history.append(position.copy())
        
            # Supprimer de la liste active
            del self.positions[symbol]
        
            # Mise à jour des métriques
            self.metrics['active_positions'] = len(self.positions)
            self.metrics['closed_positions'] += 1
        
            # Mise à jour win rate
            total_positions = self.position_stats['win_count'] + self.position_stats['loss_count']
            if total_positions > 0:
                self.metrics['win_rate'] = self.position_stats['win_count'] / total_positions
            
            # Mise à jour durée moyenne
            if self.position_stats['position_count'] > 0:
                self.metrics['avg_position_duration'] = self.position_stats['total_duration'] / self.position_stats['position_count']
        
            # Log détaillé
            self.logger.info(f"Position fermée: {symbol} à {exit_price}, PnL: {pnl:.4f}, raison: {reason}")
        
            return pnl
            
        except Exception as e:
            self.logger.error(f"Erreur fermeture position {symbol}: {str(e)}")
            return 0.0

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne les détails d'une position spécifique."""
        return self.positions.get(symbol)
    
    def get_total_pnl(self) -> float:
        """Retourne le PnL total (réalisé + non réalisé)."""
        return self.metrics['total_pnl']

    def get_all_positions(self) -> List[Dict]:
        """Retourne toutes les positions actives."""
        return list(self.positions.values())

    def get_total_exposure(self) -> float:
        """Calcule l'exposition totale du portefeuille."""
        try:
            exposure = sum(p['size'] * p['current_price'] for p in self.positions.values())
            return exposure
        except Exception as e:
            self.logger.error(f"Erreur calcul exposition: {str(e)}")
            return 0.0

    def get_market_exposure(self, market_type: str) -> float:
        """Calcule l'exposition pour un type de marché spécifique."""
        try:
            exposure = sum(
                p['size'] * p['current_price'] 
                for p in self.positions.values() 
                if p['market_type'] == market_type
            )
            return exposure
        except Exception as e:
            self.logger.error(f"Erreur calcul exposition {market_type}: {str(e)}")
            return 0.0

    def get_position_count(self) -> int:
        """Retourne le nombre de positions actives."""
        return len(self.positions)

    def calculate_drawdown(self, current_capital: float, peak_capital: float) -> float:
        """Calcule le drawdown courant."""
        try:
            if peak_capital <= 0:
                return 0.0
                
            drawdown = (peak_capital - current_capital) / peak_capital
            self.metrics['current_drawdown'] = drawdown
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
            
            return drawdown
        except Exception as e:
            self.logger.error(f"Erreur calcul drawdown: {str(e)}")
            return 0.0

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calcule le ratio de Sharpe."""
        try:
            import numpy as np
            
            if not returns or len(returns) < 2:
                return 0.0
                
            returns_array = np.array(returns)
            excess_returns = returns_array - risk_free_rate
            
            # Éviter division par zéro
            std_dev = np.std(returns_array)
            if std_dev == 0:
                return 0.0
                
            sharpe = np.mean(excess_returns) / std_dev
            self.metrics['sharpe_ratio'] = sharpe
            
            return sharpe
        except Exception as e:
            self.logger.error(f"Erreur calcul Sharpe: {str(e)}")
            return 0.0

    def get_metrics(self) -> Dict:
        """Retourne toutes les métriques de performance."""
        return self.metrics
    
    def get_total_pnl(self) -> float:
        """Retourne le PnL total (réalisé + non réalisé)."""
        return self.metrics['total_pnl']

    def reset_metrics(self):
        """Réinitialise les métriques."""
        self.metrics = {
            'total_pnl': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'active_positions': 0,
            'closed_positions': 0,
            'opened_positions': 0,
            'win_rate': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'avg_position_duration': 0.0,
            'sharpe_ratio': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0
        }
        
        self.position_stats = {
            'win_count': 0,
            'loss_count': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'total_duration': 0,
            'position_count': 0
        }
        
        self.logger.info("Métriques réinitialisées")

    def _update_metrics(self):
        """Met à jour les métriques de trading."""
        try:
            # Synchronisation explicite du PnL avec le PositionManager
            if hasattr(self, "position_manager") and self.position_manager:
                self.metrics["performance"]["total_pnl"] = self.position_manager.get_total_pnl()
        
            # Calcul win rate
            total_trades = (
                self.metrics["trades"]["winners"] + self.metrics["trades"]["losers"]
            )
            if total_trades > 0:
                self.metrics["performance"]["win_rate"] = (
                    self.metrics["trades"]["winners"] / total_trades * 100
                )

            # Mise à jour drawdown
            self._update_drawdown()

        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des métriques: {str(e)}")

    def export_history(self, filename: str) -> bool:
        """Exporte l'historique des positions en JSON."""
        try:
            with open(filename, 'w') as f:
                json.dump(
                    {
                        'position_history': self.position_history,
                        'metrics': self.metrics,
                        'position_stats': self.position_stats
                    },
                    f,
                    indent=4,
                    default=str
                )
            self.logger.info(f"Historique exporté vers {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur export historique: {str(e)}")
            return False

    async def close_all_positions(self, reason: str = "manual_close_all") -> float:
        """Ferme toutes les positions ouvertes."""
        try:
            total_pnl = 0.0
            symbols = list(self.positions.keys())
            
            for symbol in symbols:
                position = self.positions[symbol]
                exit_price = position['current_price']
                pnl = await self.close_position(symbol, exit_price, reason)
                total_pnl += pnl
                
            self.logger.info(f"Toutes les positions fermées: {len(symbols)} positions, PnL: {total_pnl:.4f}")
            return total_pnl
        except Exception as e:
            self.logger.error(f"Erreur fermeture de toutes les positions: {str(e)}")
            return 0.0

    async def check_positions_timeout(self) -> int:
        """Vérifie et ferme les positions ayant dépassé leur timeout."""
        try:
            closed_count = 0
            for symbol, position in list(self.positions.items()):
                if position['timeout'] and datetime.now() > position['timeout']:
                    self.logger.info(f"Position {symbol} a dépassé son timeout. Fermeture auto.")
                    await self.close_position(symbol, position['current_price'], 'timeout')
                    closed_count += 1
            return closed_count
        except Exception as e:
            self.logger.error(f"Erreur vérification timeouts: {str(e)}")
            return 0