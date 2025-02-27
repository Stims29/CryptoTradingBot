#!/usr/bin/env python
# tests/test_scalping.py

import os
import asyncio
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import Dict

from src.trading.bot import TradingBot
from src.trading.market_data import MarketDataManager
from src.trading.position_manager import PositionManager
from src.trading.order_manager import OrderManager
from src.trading.risk_manager import RiskManager
from src.trading.hybrid_strategy import HybridStrategy
from src.utils.logging import DetailedLogger

class ScalpingTest:
    def __init__(self, duration_minutes: int, initial_capital: float = 10.0):
        self.duration_minutes = duration_minutes
        self.initial_capital = initial_capital
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        # Loggers
        self.main_logger = DetailedLogger("main")
        self.bot_logger = DetailedLogger("bot")
        
        # État du test
        self.bot = None
        self.metrics_history = []
        self.error_count = 0
        self.last_metrics_update = datetime.now()
        
        # Configuration du monitoring
        self.monitor_config = {
            'metrics_interval': 2,  # Secondes
            'save_interval': 30,    # Secondes
            'detailed_logging': True,
            'monitor_indicators': True
        }
        
        # Dossier résultats
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    async def setup_bot(self) -> TradingBot:
        """Configure et initialise le bot avec paramètres optimisés."""
        try:
            # Création du bot avec capital initial
            bot = TradingBot(simulation_mode=True)
            bot.current_capital = self.initial_capital
            bot.peak_capital = self.initial_capital
            
            # Configuration explicite des symboles
            bot.symbols = [
                "BTC/USDT", #"ETH/USDT", #"XRP/USDT",
                #"DOT/USDT", "ADA/USDT", 
                #"SOL/USDT", #"AVAX/USDT",
                #"UNI/USDT", "AAVE/USDT",
                #"LINK/USDT", "ETH/BTC"
            ]
            
            # Log de vérification des symboles
            #self.main_logger.info(f"Configuration du bot avec {len(bot.symbols)} symboles: {', '.join(bot.symbols)}")
            self.main_logger.info(f"Configuration du bot avec symbole: {bot.symbols[0]}")
        
            # Initialisation des composants avec leur propre logger
            bot.market_data = MarketDataManager(simulation_mode=True)
            bot.market_data.logger = self.bot_logger
            
            bot.position_manager = PositionManager(simulation_mode=True, initial_capital=self.initial_capital)
            bot.position_manager.logger = self.bot_logger
            
            bot.order_manager = OrderManager(simulation_mode=True,)
            bot.order_manager.logger = self.bot_logger
            
            bot.risk_manager = RiskManager(simulation_mode=True, initial_capital=self.initial_capital)
            bot.risk_manager.logger = self.bot_logger
            
            bot.strategy = HybridStrategy()
            bot.strategy.logger = self.bot_logger

            bot.strategy.market_data = bot.market_data  # Donner accès à market_data

            # Activation du logging détaillé pour la stratégie
            bot.strategy._verbose_logging = True
            
            # Assignation du logger principal
            bot.logger = self.bot_logger

            # Log de vérification des symboles
            self.main_logger.info(f"Configuration du bot avec {len(bot.symbols)} symboles: {', '.join(bot.symbols)}")
            
            # Vérifications explicites
            if not bot.market_data or not bot.position_manager or not bot.order_manager or not bot.risk_manager or not bot.strategy:
                raise ValueError("Un ou plusieurs composants n'ont pas été initialisés correctement")
                
            if not bot.symbols:
                raise ValueError("Aucun symbole configuré pour le bot")
                
            self.main_logger.info(f"Bot configuré avec succès: capital={self.initial_capital}€, symboles={len(bot.symbols)}")
            
            return bot
            
        except Exception as e:
            self.main_logger.error(f"Erreur setup bot: {e}")
            raise

    async def monitor_performance(self) -> None:
        """Monitore les performances du bot en temps réel avec protection contre les erreurs."""
        try:
            while datetime.now() < self.end_time and self.bot and self.bot._running:
                try:
                    current_time = datetime.now()
                    
                    # Métriques temps réel avec protection contre les attributs manquants
                    metrics = {
                        'timestamp': current_time.isoformat(),
                        'remaining_time': str(self.end_time - current_time),
                        'current_capital': self.bot.current_capital if hasattr(self.bot, 'current_capital') else 0.0,
                    }
                    
                    # Position manager metrics (avec protection)
                    if hasattr(self.bot, 'position_manager'):
                        pm = self.bot.position_manager
                        
                        # Total PnL avec vérification
                        if hasattr(pm, 'metrics') and 'total_pnl' in pm.metrics:
                            metrics['total_pnl'] = round(pm.metrics['total_pnl'], 4)
                        else:
                            metrics['total_pnl'] = 0.0
                            
                        # Active positions avec vérification
                        if hasattr(pm, 'positions'):
                            metrics['active_positions'] = len(pm.positions)
                        else:
                            metrics['active_positions'] = 0
                    else:
                        metrics['total_pnl'] = 0.0
                        metrics['active_positions'] = 0
                    
                    # Order manager et Strategy metrics (avec protection)
                    if hasattr(self.bot, 'order_manager') and hasattr(self.bot.order_manager, 'order_count'):
                        metrics['trades_executed'] = self.bot.order_manager.order_count
                    else:
                        metrics['trades_executed'] = 0
                        
                    if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'metrics'):
                        metrics['signals_generated'] = getattr(self.bot.strategy.metrics, 'generated', 0)
                    else:
                        metrics['signals_generated'] = 0
                        
                    # Win rate (avec protection)
                    if hasattr(self.bot, 'position_manager') and hasattr(self.bot.position_manager, 'metrics') and 'win_rate' in self.bot.position_manager.metrics:
                        metrics['win_rate'] = round(self.bot.position_manager.metrics['win_rate'] * 100, 2)
                    else:
                        metrics['win_rate'] = 0.0
                    
                    # Métriques détaillées si activé
                    if self.monitor_config['detailed_logging']:
                        # Drawdown (avec protection)
                        if hasattr(self.bot, 'position_manager') and hasattr(self.bot.position_manager, 'metrics') and 'current_drawdown' in self.bot.position_manager.metrics:
                            metrics['drawdown'] = round(self.bot.position_manager.metrics['current_drawdown'] * 100, 2)
                        else:
                            metrics['drawdown'] = 0
                            
                        # Exposition (avec protection)
                        if hasattr(self.bot, 'position_manager') and hasattr(self.bot.position_manager, 'get_total_exposure'):
                            metrics['exposure'] = round(self.bot.position_manager.get_total_exposure(), 2)
                        else:
                            metrics['exposure'] = 0
                            
                        # Signaux rejetés (avec protection)
                        if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'metrics'):
                            metrics['rejected_signals'] = getattr(self.bot.strategy.metrics, 'rejected', 0)
                        else:
                            metrics['rejected_signals'] = 0
                            
                        metrics['error_count'] = self.error_count
                    
                    # Indicateurs si activés
                    if self.monitor_config['monitor_indicators'] and hasattr(self.bot, 'market_data') and hasattr(self.bot, 'strategy'):
                        for symbol in self.bot.symbols[:3]:  # Top 3 seulement
                            try:
                                market_data = await self.bot.market_data.get_market_data(symbol)
                                if market_data is not None and not market_data.empty and hasattr(self.bot.strategy, '_calculate_indicators'):
                                    metrics[f"{symbol}_indicators"] = self.bot.strategy._calculate_indicators(market_data)
                            except Exception as e:
                                self.main_logger.error(f"Erreur récupération indicateurs {symbol}: {e}")
                    
                    # Log et sauvegarde
                    self.metrics_history.append(metrics)
                    self.bot_logger.info(f"METRICS: {json.dumps(metrics, default=str)}")
                    
                    # Sauvegarde périodique
                    if (current_time - self.last_metrics_update).seconds >= self.monitor_config['save_interval']:
                        await self.save_metrics()
                        self.last_metrics_update = current_time
                    
                    await asyncio.sleep(self.monitor_config['metrics_interval'])
                    
                except Exception as e:
                    self.main_logger.error(f"Erreur monitoring cycle: {e}")
                    self.error_count += 1
                    await asyncio.sleep(1)  # Pause en cas d'erreur
                    
        except asyncio.CancelledError:
            self.main_logger.warning("Monitoring interrompu")
        except Exception as e:
            self.main_logger.error(f"Erreur monitoring: {e}")
            self.error_count += 1

    async def save_metrics(self) -> None:
        """Sauvegarde les métriques de performance."""
        try:
            if not self.metrics_history:
                self.main_logger.warning("Aucune métrique à sauvegarder")
                return
                
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_df.to_csv(
                os.path.join(self.results_dir, f"metrics_{self.session_id}.csv"),
                index=False
            )
            self.main_logger.info(f"Métriques sauvegardées: {len(self.metrics_history)} entrées")
        except Exception as e:
            self.main_logger.error(f"Erreur sauvegarde métriques: {e}")

    async def save_final_results(self, duration: timedelta) -> None:
        """Sauvegarde les résultats finaux du test avec gestion des erreurs."""
        try:
            # Préparation des informations de test
            test_info = {
                "session_id": self.session_id,
                "duration": str(duration),
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "initial_capital": self.initial_capital
            }
            
            # Préparation des métriques avec vérifications
            performance = {
                "final_capital": 0.0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
            
            signals = {
                "generated": 0,
                "executed": 0,
                "rejected": 0,
                "errors": self.error_count
            }
            
            strategy_breakdown = {
                "BREAKOUT": 0,
                "MEAN_REVERSION": 0,
                "MOMENTUM": 0,
                "ORDER_BOOK": 0
            }
            
            # Récupération des données si disponibles
            if self.bot:
                # Capital final
                if hasattr(self.bot, 'current_capital'):
                    performance["final_capital"] = self.bot.current_capital
                
                # Position manager metrics
                if hasattr(self.bot, 'position_manager') and hasattr(self.bot.position_manager, 'metrics'):
                    pm_metrics = self.bot.position_manager.metrics
                    performance["total_pnl"] = pm_metrics.get('total_pnl', 0.0)
                    performance["win_rate"] = pm_metrics.get('win_rate', 0.0)
                    performance["max_drawdown"] = pm_metrics.get('max_drawdown', 0.0)
                
                # Order manager
                if hasattr(self.bot, 'order_manager'):
                    performance["total_trades"] = getattr(self.bot.order_manager, 'order_count', 0)
                
                # Strategy metrics
                if hasattr(self.bot, 'strategy') and hasattr(self.bot.strategy, 'metrics'):
                    strategy_metrics = self.bot.strategy.metrics
                    signals["generated"] = getattr(strategy_metrics, 'generated', 0)
                    signals["executed"] = getattr(strategy_metrics, 'executed', 0)
                    signals["rejected"] = getattr(strategy_metrics, 'rejected', 0)
                    
                    # Strategy breakdown
                    if hasattr(strategy_metrics, 'strategy_signals'):
                        strategy_breakdown = strategy_metrics.strategy_signals
            
            # Construction du résultat final
            final_metrics = {
                "test_info": test_info,
                "performance": performance,
                "signals": signals,
                "strategy_breakdown": strategy_breakdown
            }
            
            # Sauvegarde JSON
            with open(os.path.join(self.results_dir, f"results_{self.session_id}.json"), 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            # Log résultats
            self.main_logger.info("TEST COMPLETED")
            self.main_logger.info(f"Résultats: {json.dumps(final_metrics, indent=2, default=str)}")
            
        except Exception as e:
            self.main_logger.error(f"Erreur sauvegarde résultats: {e}")

    async def run(self) -> Dict:
        """Exécute le test de scalping avec gestion robuste des erreurs et arrêt propre."""
        start_time = datetime.now()
        bot_task = None
        monitor_task = None
    
        try:
            # Setup et configuration
            self.main_logger.info(f"Démarrage test scalping - Durée: {self.duration_minutes}min")
            self.bot = await self.setup_bot()
        
            # Vérifications de base
            if not self.bot or not isinstance(self.bot, TradingBot):
                self.main_logger.error("Échec d'initialisation du bot")
                return {}

            # Création d'un event pour la synchronisation
            stop_event = asyncio.Event()
        
            # Création des tâches avec protection
            try:
                # Démarrage des tâches principales
                bot_task = asyncio.create_task(self.bot.start())
                monitor_task = asyncio.create_task(self.monitor_performance())
            
                # Tâche de timer pour l'arrêt
                async def stop_timer():
                    await asyncio.sleep(self.duration_minutes * 60)
                    stop_event.set()
                
                timer_task = asyncio.create_task(stop_timer())
            
                # Attendre soit le timer, soit une interruption
                try:
                    await stop_event.wait()
                    self.main_logger.info("Durée du test atteinte, arrêt en cours...")
                except asyncio.CancelledError:
                    self.main_logger.info("Test interrompu, arrêt en cours...")
                finally:
                    # Annulation ordonnée des tâches
                    tasks = [t for t in (bot_task, monitor_task, timer_task) if t and not t.done()]
                
                    # Annuler toutes les tâches
                    for task in tasks:
                        task.cancel()
                
                    # Attendre leur terminaison avec timeout
                    if tasks:
                        try:
                            await asyncio.wait(tasks, timeout=5)
                        except asyncio.TimeoutError:
                            self.main_logger.warning("Timeout lors de l'arrêt des tâches")
                        except Exception as e:
                            self.main_logger.error(f"Erreur lors de l'arrêt des tâches: {e}")
                
            except Exception as e:
                self.main_logger.error(f"Erreur pendant l'exécution: {e}")
            finally:
                # Arrêt explicite du bot
                if self.bot and hasattr(self.bot, 'stop'):
                    try:
                        await asyncio.wait_for(self.bot.stop(), timeout=5)
                    except asyncio.TimeoutError:
                        self.main_logger.error("Timeout lors de l'arrêt du bot")
                    except Exception as e:
                        self.main_logger.error(f"Erreur lors de l'arrêt du bot: {e}")
        
            # Sauvegarde des résultats
            duration = datetime.now() - start_time
            await self.save_final_results(duration)
        
            # Retour des métriques
            return self.bot.get_metrics() if self.bot and hasattr(self.bot, 'get_metrics') else {}
        
        except Exception as e:
            self.main_logger.error(f"Erreur critique du test: {e}")
            return {}
        finally:
            # Nettoyage final avec protection supplémentaire
            try:
                if self.bot and hasattr(self.bot, '_running') and self.bot._running:
                    await asyncio.wait_for(self.bot.stop(), timeout=3)
            except Exception as e:
                self.main_logger.error(f"Erreur lors du nettoyage final: {e}")
        
            self.main_logger.info("Test terminé")

async def main():
    """Point d'entrée principal avec gestion robuste des erreurs."""
    try:
        # Configuration des paramètres
        duration_minutes = 15  
        initial_capital = 10.0
        
        # Exécution test
        test = ScalpingTest(duration_minutes, initial_capital)
        await test.run()
    except Exception as e:
        print(f"ERREUR CRITIQUE dans main(): {e}")

if __name__ == "__main__":
    asyncio.run(main())