#!/usr/bin/env python
# src/trading/risk_manager.py

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np


class RiskManager:
    def __init__(self, simulation_mode: bool = True, initial_capital: float = 100.0):
        self.logger = logging.getLogger(__name__)
        self.simulation_mode = simulation_mode
        self.initial_capital = initial_capital

        # Configuration par catégorie de marché ajustée pour scalping 
        self.market_configs = {
            'MAJOR': {
                'allocation_max': 0.15,    # 15% inchangé
                'position_max': 0.05,      # 5% inchangé
                'stop_loss': 0.002,        # Modifié à 0.2%
                'take_profit': 0.004,      # Modifié à 0.4%
                'trailing_stop': 0.001,    # 0.1% inchangé
                'partial_tp': [            # Sorties partielles
                    {'level': 0.002, 'size': 0.5},  # 50% à 0.2%
                    {'level': 0.004, 'size': 0.5}   # 50% à 0.4%
                ]
            },
            'ALTCOINS': {
                'allocation_max': 0.15,    # Maintenu
                'position_max': 0.05,      # Maintenu
                'stop_loss': 0.002,        # Modifié à 0.2%
                'take_profit': 0.004,      # Modifié à 0.4%
                'trailing_stop': 0.001,    # 0.1%
                'partial_tp': [
                    {'level': 0.002, 'size': 0.5},
                    {'level': 0.004, 'size': 0.5}
                ]
            },
            'DEFI': {
                'allocation_max': 0.1,
                'position_max': 0.05,
                'stop_loss': 0.002,        # Modifié à 0.2%
                'take_profit': 0.004,      # Modifié à 0.4%
                'trailing_stop': 0.001,          # 0.1% trailing stop
                'partial_tp': [                  # Sorties partielles
                    {'level': 0.002, 'size': 0.5},  # 50% à 0.2%
                    {'level': 0.004, 'size': 0.5}   # 50% à 0.4%
                ]
            },
            'NFT_METAVERSE': {
                'allocation_max': 0.1,
                'position_max': 0.05,
                'stop_loss': 0.002,        # Modifié à 0.2%
                'take_profit': 0.004,      # Modifié à 0.4%
                'trailing_stop': 0.001,          # 0.1% trailing stop
                'partial_tp': [                  # Sorties partielles
                    {'level': 0.002, 'size': 0.5},  # 50% à 0.2%
                    {'level': 0.004, 'size': 0.5}   # 50% à 0.4%
                ]
            },
            'BTC_PAIRS': {
                'allocation_max': 0.05,
                'position_max': 0.03,
                'stop_loss': 0.002,        # Modifié à 0.2%
                'take_profit': 0.004,      # Modifié à 0.4%
                'trailing_stop': 0.001,          # 0.1% trailing stop
                'partial_tp': [                  # Sorties partielles
                    {'level': 0.002, 'size': 0.5},  # 50% à 0.2%
                    {'level': 0.004, 'size': 0.5}   # 50% à 0.4%
                ]
            }
        }

        # Configuration générale optimisée pour scalping
        self.config = {
            'max_positions': 3,            # Augmenté pour permettre plus de positions simultanées
            'max_trades_per_hour': 30,     # 
            'min_trade_interval': 1,       # Intervalles de 1 seconde
            'max_daily_loss': -0.15,       # Augmenté à 15% pour le test
            'max_drawdown': 0.1,           # Drawdown maximum à 10%
            'profit_lock': 0.001,          # Sécuriser les profits dès 0.1%
            'min_volume': 10000,           # Seuil de volume minimum réduit
            'max_spread': 0.003,           # Spread maximum autorisé à 0.3%
            'slippage_tolerance': 0.001,   # Tolérance de slippage à 0.1%
            'volatility_filter': {
                'threshold': 0.001,        # Seuil de volatilité minimum réduit
                'timeframe': 300           # 5 minutes
            },
            'dynamic_sizing': True,        # Maintenir le sizing dynamique
            'risk_reduction': {
                'consecutive_losses': 1,    # Réduction après 1 seule perte
                'reduction_factor': 0.5     # Réduction de 50%
            }
        }

        # État du risk management
        self.positions: Dict[str, Dict] = {}
        self.trades_history: List[Dict] = []
        self._last_trade_time = datetime.now()
        self._trades_this_hour = 0
        self._hourly_reset_time = datetime.now()
        self._consecutive_losses = 0
        self._risk_factor = 1.0

        # Métriques avancées
        self.metrics = {
            "total_exposure": 0.0,
            "largest_position": 0.0,
            "daily_pnl": 0.0,
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "locked_profit": 0.0,
            "trades_count": 0,
            "rejected_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "risk_reward_ratio": 0.0,
            "sharpe_ratio": 0.0,
        }

        self.logger.info(
            f"RiskManager optimisé pour scalping initialisé\n"
            f"Mode simulation: {simulation_mode}\n"
            f"Max positions: {self.config['max_positions']}\n"
            f"Max trades/heure: {self.config['max_trades_per_hour']}\n"
            f"Intervalle min: {self.config['min_trade_interval']}s"
        )

    def get_market_category(self, symbol: str) -> str:
        """Détermine la catégorie de marché pour un symbole."""
        # MAJOR - Cryptomonnaies majeures à volume très élevé
        if symbol in ['BTC/USDT']: # , 'ETH/USDT' , 'XRP/USDT'  
            return 'MAJOR'
    
        # ALTCOINS - Altcoins établis à volume élevé
        #elif symbol in ['SOL/USDT']: # 'DOT/USDT', 'ADA/USDT', 'AVAX/USDT
        #    return 'ALTCOINS'
    
        # DEFI - Tokens de finance décentralisée
        #elif symbol in ['UNI/USDT', 'AAVE/USDT', 'LINK/USDT', 'SUSHI/USDT', 'CRV/USDT']:
        #    return 'DEFI'
    
        # NFT_METAVERSE - Tokens liés aux NFT et au metaverse
        #elif symbol in ['SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'ENJ/USDT']:
        #    return 'NFT_METAVERSE'
    
        # BTC_PAIRS - Paires contre BTC
        #elif symbol in ['ETH/BTC', 'ADA/BTC', 'XRP/BTC', 'DOT/BTC']:
        #    return 'BTC_PAIRS'
    
        # Catégorie par défaut si le symbole n'est pas reconnu
        else:
            self.logger.warning(f"Symbole non catégorisé: {symbol}, classé comme ALTCOINS par défaut")
            return 'ALTCOINS'
       

    def check_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Vérification plus permissive pour scalping haute fréquence."""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                return False, "Symbole manquant"

            category = self.get_market_category(symbol)
        
            # Désactivation temporaire des vérifications d'exposition pour le Test 3C
            self.logger.info(f"Test 3C: Vérification d'exposition désactivée pour {symbol}")
        
            # 5. Vérification intervalle minimum
            time_since_last = (datetime.now() - self._last_trade_time).total_seconds()
            if time_since_last < self.config["min_trade_interval"]:
                return False, f"Intervalle min non respecté: {time_since_last:.2f}s"

            # 6. Vérification nombre max de trades par heure
            if self._trades_this_hour >= self.config["max_trades_per_hour"]:
                return False, f"Max trades par heure atteint: {self._trades_this_hour}"

            # 7. Mise à jour compteurs
            self._trades_this_hour += 1
            self._last_trade_time = datetime.now()

            # Mode test: acceptation du signal
            return True, "Trade autorisé pour Test 3C"

        except Exception as e:
            self.logger.error(f"Erreur vérification trade: {str(e)}")
            return False, f"Erreur interne: {str(e)}"

    def calculate_position_size(self, capital: float, price: float, symbol: str = None) -> Optional[float]:
        """Calcule une taille de position minimale pour le Test 3C."""
        try:
            # Position ultra-minimaliste pour test (0.1% du capital)
            base_size = capital * 0.001
            quantity = base_size / price if price > 0 else 0
        
            # Toujours retourner une valeur positive minimale
            return max(0.0001, quantity)
        
        except Exception as e:
            self.logger.error(f"Erreur calcul taille position: {e}")
            return 0.0001  # Valeur minimale garantie

    def get_position_size(
        self, capital: float, price: float, symbol: str
    ) -> Optional[float]:
        """Calcule la taille de position optimale avec ajustements dynamiques."""
        try:
            if price <= 0 or capital <= 0:
                return None

            category = self.get_market_category(symbol)
            market_config = self.market_configs[category]

            # Taille de base selon catégorie
            base_size = capital * market_config["position_max"]

            # Ajustement dynamique
            if self.config["dynamic_sizing"]:
                # Facteur positions actives
                position_factor = 1 - (
                    len(self.positions) / self.config["max_positions"]
                )
                base_size *= max(0.3, position_factor)

                # Facteur drawdown
                if self.metrics["current_drawdown"] > 0:
                    drawdown_factor = 1 - (
                        self.metrics["current_drawdown"] / self.config["max_drawdown"]
                    )
                    base_size *= max(0.5, drawdown_factor)

                # Facteur pertes consécutives
                if (
                    self._consecutive_losses
                    >= self.config["risk_reduction"]["consecutive_losses"]
                ):
                    base_size *= self.config["risk_reduction"]["reduction_factor"]

            # Conversion en quantité
            quantity = base_size / price

            # Limites min/max
            min_quantity = 0.001  # Minimum technique
            max_quantity = market_config["position_max"] * capital / price
            quantity = max(min_quantity, min(quantity, max_quantity))

            return round(quantity, 8)

        except Exception as e:
            self.logger.error(f"Erreur calcul taille position: {str(e)}")
            return None

    def update_position(self, position_data: Dict) -> None:
        """Met à jour une position avec gestion avancée des métriques."""
        try:
            symbol = position_data.get("symbol")
            if not symbol:
                return

            category = self.get_market_category(symbol)

            # Calcul PnL si position existante
            if symbol in self.positions:
                old_position = self.positions[symbol]
                pnl = self._calculate_position_pnl(old_position, position_data)

                if (
                    old_position["status"] == "CLOSED"
                    and position_data["status"] == "CLOSED"
                ):
                    # Position fermée, mise à jour métriques
                    self._update_metrics(pnl, category)

                    # Mise à jour historique
                    trade_record = {
                        "symbol": symbol,
                        "category": category,
                        "entry_price": old_position["entry_price"],
                        "exit_price": position_data["current_price"],
                        "size": old_position["size"],
                        "pnl": pnl,
                        "duration": position_data["close_time"]
                        - old_position["open_time"],
                        "timestamp": datetime.now(),
                    }
                    self.trades_history.append(trade_record)

                    # Suppression position fermée
                    del self.positions[symbol]
                else:
                    # Mise à jour position existante
                    self.positions[symbol].update(position_data)
            else:
                # Nouvelle position
                self.positions[symbol] = position_data

            # Mise à jour métriques globales
            self._update_exposure_metrics()

        except Exception as e:
            self.logger.error(f"Erreur mise à jour position: {str(e)}")

    def _calculate_position_pnl(self, old_pos: Dict, new_pos: Dict) -> float:
        """Calcule le PnL d'une position."""
        try:
            price_diff = new_pos["current_price"] - old_pos["entry_price"]
            direction = 1 if old_pos["side"] == "buy" else -1
            return price_diff * old_pos["size"] * direction
        except Exception as e:
            self.logger.error(f"Erreur calcul PnL: {str(e)}")
            return 0.0

    def _update_metrics(self, pnl: float, category: str) -> None:
        """Met à jour toutes les métriques après un trade."""
        try:
            # Mise à jour PnL et drawdown
            self.metrics["daily_pnl"] += pnl
            self._update_drawdown(pnl)

            # Mise à jour compteurs profits/pertes
            if pnl > 0:
                self._consecutive_losses = 0
                self.metrics["avg_win"] = (
                    self.metrics["avg_win"] * self.metrics["trades_count"] + pnl
                ) / (self.metrics["trades_count"] + 1)
            else:
                self._consecutive_losses += 1
                self.metrics["avg_loss"] = (
                    self.metrics["avg_loss"] * self.metrics["trades_count"] + abs(pnl)
                ) / (self.metrics["trades_count"] + 1)

            # Mise à jour statistiques
            self.metrics["trades_count"] += 1
            profits = [t["pnl"] for t in self.trades_history if t["pnl"] > 0]
            losses = [abs(t["pnl"]) for t in self.trades_history if t["pnl"] < 0]

            if profits:
                total_profit = sum(profits)
                self.metrics["win_rate"] = len(profits) / self.metrics["trades_count"]
            else:
                total_profit = 0

            total_loss = sum(losses) if losses else 0
            self.metrics["profit_factor"] = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )

            if self.metrics["avg_loss"] > 0:
                self.metrics["risk_reward_ratio"] = (
                    self.metrics["avg_win"] / self.metrics["avg_loss"]
                )

            # Calcul Sharpe Ratio
            if len(self.trades_history) > 1:
                returns = [t["pnl"] for t in self.trades_history]
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    self.metrics["sharpe_ratio"] = (
                        mean_return / std_return * np.sqrt(252)
                    )  # Annualisé
        except Exception as e:
            self.logger.error(f"Erreur mise à jour métriques: {str(e)}")

    def _calculate_total_exposure(self) -> float:
        """Calcule l'exposition totale actuelle du portefeuille."""
        try:
            return sum(
                pos.get("size", 0) * pos.get("current_price", 0)
                for pos in self.positions.values()
            )
        except Exception as e:
            self.logger.error(f"Erreur calcul exposition totale: {str(e)}")
            return 0.0

    def _calculate_category_exposure(self, category: str) -> float:
        """Calcule l'exposition totale pour une catégorie."""
        try:
            return sum(
                pos.get("size", 0) * pos.get("current_price", 0)
                for pos in self.positions.values()
                if self.get_market_category(pos.get("symbol", "")) == category
            )
        except Exception as e:
            self.logger.error(f"Erreur calcul exposition {category}: {str(e)}")
            return 0.0

    def _update_exposure_metrics(self) -> None:
        """Met à jour les métriques d'exposition."""
        try:
            # Exposition totale
            total_exposure = self._calculate_total_exposure()
            self.metrics["total_exposure"] = total_exposure

            # Plus grande position
            if self.positions:
                largest = max(
                    pos.get("size", 0) * pos.get("current_price", 0)
                    for pos in self.positions.values()
                )
                self.metrics["largest_position"] = largest

        except Exception as e:
            self.logger.error(f"Erreur update métriques exposition: {str(e)}")

    def _update_drawdown(self, pnl: float) -> None:
        """Met à jour les métriques de drawdown."""
        try:
            if pnl < 0:
                self.metrics["current_drawdown"] += abs(pnl) / self.initial_capital
                self.metrics["max_drawdown"] = max(
                    self.metrics["max_drawdown"], self.metrics["current_drawdown"]
                )
            else:
                # Réduction du drawdown avec les gains
                self.metrics["current_drawdown"] = max(
                    0.0, self.metrics["current_drawdown"] - pnl / self.initial_capital
                )

                # Mise à jour profit lock
                profit_to_lock = min(pnl, self.config["profit_lock"] * abs(pnl))
                self.metrics["locked_profit"] += profit_to_lock

        except Exception as e:
            self.logger.error(f"Erreur update drawdown: {str(e)}")

    def get_metrics(self) -> Dict:
        """Retourne les métriques actuelles."""
        return self.metrics.copy()

    def get_config(self) -> Dict:
        """Retourne la configuration actuelle."""
        return {
            "market_configs": self.market_configs.copy(),
            "general_config": self.config.copy(),
        }

    def get_trade_stats(self) -> Dict:
        """Retourne les statistiques détaillées de trading."""
        try:
            if not self.trades_history:
                return {}

            stats = {
                "total_trades": len(self.trades_history),
                "winning_trades": len([t for t in self.trades_history if t["pnl"] > 0]),
                "losing_trades": len([t for t in self.trades_history if t["pnl"] < 0]),
                "avg_trade_duration": sum(
                    (t["duration"] for t in self.trades_history), timedelta()
                )
                / len(self.trades_history),
                "best_trade": max(self.trades_history, key=lambda x: x["pnl"])["pnl"],
                "worst_trade": min(self.trades_history, key=lambda x: x["pnl"])["pnl"],
                "consecutive_losses": self._consecutive_losses,
                "current_risk_factor": self._risk_factor,
            }

            # Stats par catégorie
            stats["category_performance"] = {}
            for category in self.market_configs.keys():
                cat_trades = [
                    t for t in self.trades_history if t["category"] == category
                ]
                if cat_trades:
                    stats["category_performance"][category] = {
                        "trades": len(cat_trades),
                        "win_rate": len([t for t in cat_trades if t["pnl"] > 0])
                        / len(cat_trades),
                        "total_pnl": sum(t["pnl"] for t in cat_trades),
                        "avg_pnl": sum(t["pnl"] for t in cat_trades) / len(cat_trades),
                    }

            return stats

        except Exception as e:
            self.logger.error(f"Erreur calcul statistiques: {str(e)}")
            return {}
