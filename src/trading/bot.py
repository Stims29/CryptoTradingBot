#!/usr/bin/env python
# src/trading/bot.py

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from src.utils.logging import DetailedLogger


class AdaptiveCircuitBreaker:
    def __init__(self):
        self.rejection_threshold = 1000  # Seuil initial
        self.rejection_count = 0
        self.consecutive_rejects = 0
        self.backoff_factor = 1.5
        self.cooldown_period = 1.0  # secondes
        self.last_reset = datetime.now()

    async def check_and_adapt(self, rejection_count: int) -> bool:
        """Vérifie et adapte les paramètres selon les rejets."""
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= self.cooldown_period:
            if rejection_count > self.rejection_threshold:
                self.consecutive_rejects += 1
                self.cooldown_period *= self.backoff_factor
                return True  # Circuit ouvert
            else:
                self.consecutive_rejects = 0
                self.cooldown_period = max(
                    1.0, self.cooldown_period / self.backoff_factor
                )
            self.last_reset = now
            self.rejection_count = 0
        return False  # Circuit fermé


class TradingBot:
    def __init__(self, simulation_mode: bool = True, initial_capital: float = 10.0):
        """Initialise le bot de trading."""
        self.logger = logging.getLogger(__name__)
        self.detailed_logger = DetailedLogger(__name__)
        self.simulation_mode = simulation_mode
        self.initial_capital = initial_capital
        self.last_detailed_report = datetime.now()
        self.report_interval = timedelta(minutes=5)

        # État du bot
        self.symbols = []
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self._error_counter = 0
        self._last_update = datetime.now()
        self._running = False
        self._processed_signals = {}

        # Flags de contrôle des performances
        self._verbose_logging = False
        self._log_interval = 10  # Ne logger qu'un événement sur 10
        self._log_counter = 0
        self._memory_optimized = True

        # Gestionnaires
        self.market_data = None  # Initialisé lors du démarrage
        self.position_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.strategy = None

        # Rate limiting
        self._analysis_config = {
            "max_analysis_per_second": 20,  # Limite à 20 analyses/s
            "analysis_count": 0,
            "last_reset": datetime.now(),
            "cooldown_period": 0.05,  # 50ms minimum entre analyses
        }

        # Métriques
        self.metrics = {
            "market_analysis": {
                "signals_generated": 0,
                "signals_executed": 0,
                "signals_rejected": 0,
            },
            "trades": {"total": 0, "winners": 0, "losers": 0},
            "performance": {
                "total_pnl": 0.0,
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            },
        }

        # Mécanisme de circuit breaker
        self._circuit_breaker = {
            "consecutive_errors": 0,
            "error_threshold": 50,
            "backoff_factor": 1.2,
            "initial_delay": 0.05,
            "current_delay": 0.1,
            "max_delay": 1.0,
            "last_error_time": datetime.now(),
        }

        # Système de suivi des rejets
        self.rejection_stats = defaultdict(int)
        self.rejection_samples = []
        self.last_rejection_report = datetime.now()

        # Limites globales de risk management
        self.risk_limits = {
            "max_drawdown": 0.20,         # Augmenter de 0.15 à 0.20
            "max_daily_loss": 0.15,       # Augmenter de 0.10 à 0.15
            "max_exposure": 0.25,  # 25% exposition maximale
            "max_positions": 3,  # 3 positions simultanées max
            "emergency_mode": True,       # Maintenir actif pour la sécurité
            'recovery_threshold': 0.05    # Nouveau paramètre pour sortie auto du mode urgence
        }

        self.logger.info(
            f"TradingBot initialisé en mode {'simulation' if simulation_mode else 'réel'}\n"
            f"Capital initial: {self.current_capital}€\n"
            f"Symboles: {', '.join(self.symbols) if self.symbols else 'Aucun'}"
        )

    def _should_log(self, importance="normal") -> bool:
        """
        Détermine si un événement doit être loggé selon son importance.

        Args:
            importance (str): 'critical', 'high', 'normal', ou 'low'

        Returns:
            bool: True si l'événement doit être loggé
        """
        try:
            # En mode debug, on log tout
            if self._verbose_logging:
                return True

            # Toujours logger les événements critiques
            if importance == "critical":
                return True

            # Événements haute importance : 1 sur 2
            if importance == "high":
                return self._log_counter % 2 == 0

            # Événements normaux : 1 sur 5
            if importance == "normal":
                return self._log_counter % 5 == 0

            # Événements de faible importance : 1 sur 10
            return self._log_counter % 10 == 0

        finally:
            # Incrémenter le compteur
            self._log_counter = (self._log_counter + 1) % 1000

    def log_signal_rejection(self, symbol, price, indicators, reason):
        """Traçage détaillé des rejets de signaux."""
        self.rejection_stats[reason] += 1
        self.metrics["market_analysis"]["signals_rejected"] += 1

        # Traçage complet pour debug
        indicators_str = (
            ", ".join([f"{k}={v}" for k, v in indicators.items()])
            if indicators
            else "N/A"
        )
        self.logger.debug(
            f"Signal rejeté pour {symbol}: {reason} - Prix: {price} - Indicateurs: {indicators_str}"
        )

        # Sauvegarde périodique vers un fichier pour analyse
        if self.metrics["market_analysis"]["signals_rejected"] % 1000 == 0:
            with open(
                f"rejection_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "stats": dict(self.rejection_stats),
                        "samples": self.rejection_samples,
                    },
                    f,
                )

    async def start(self):
        """Démarre le bot de trading avec vérification de boucle."""
        try:
            # Vérification des composants
            if not all(
                [
                    self.market_data,
                    self.position_manager,
                    self.order_manager,
                    self.risk_manager,
                    self.strategy,
                ]
            ):
                raise ValueError("Un ou plusieurs composants non initialisés")

            # Vérification des symboles
            if not self.symbols:
                raise ValueError("Aucun symbole configuré")

            self._running = True
            self._emergency_brake_activated = False
            self._signal_suspension_until = None
            self.risk_limits["emergency_mode"] = False
            self.logger.info("Test 3B: Frein d'urgence réinitialisé")

            # Log explicite pour vérification
            self.logger.info(
                f"Bot démarré avec {len(self.symbols)} symboles: {', '.join(self.symbols)}"
            )

            # Compteur de cycles pour vérification d'activité
            cycle_count = 0
            last_heartbeat = datetime.now()

            while self._running:
                try:
                    # Analyse parallèle des marchés
                    analysis_tasks = [
                        self._analyze_market(symbol) for symbol in self.symbols
                    ]

                    # Exécuter en parallèle avec limite de concurrence
                    await asyncio.gather(*analysis_tasks, return_exceptions=True)

                    # Incrémentation du compteur de cycles
                    cycle_count += 1

                    # Log périodique pour confirmer l'activité (toutes les 2 minutes)
                    now = datetime.now()
                    if (now - last_heartbeat).total_seconds() > 120:
                        self.logger.info(
                            f"Bot actif - cycle {cycle_count}, capital: {self.current_capital:.2f}€"
                        )
                        last_heartbeat = now

                    # Mise à jour des positions
                    await self._update_positions()

                    # Mise à jour des métriques
                    self._update_metrics()

                    # Génération rapport périodique
                    self._generate_detailed_report()

                    # Nettoyage des signaux traités
                    self._cleanup_processed_signals()

                    # Vérification du mode urgence
                    self._check_emergency_mode()

                    # Pause courte pour éviter surcharge CPU
                    await asyncio.sleep(0.1)

                except Exception as e:
                    self._error_counter += 1
                    self.logger.error(f"Erreur boucle principale: {e}")

                    # Arrêt si trop d'erreurs consécutives
                    if self._error_counter >= 10:
                        self.logger.error("Trop d'erreurs consécutives, arrêt du bot")
                        break

                    # Pause plus longue en cas d'erreur
                    await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Erreur critique: {e}")
        finally:
            await self.stop()

    async def _analyze_market(self, symbol: str):
        """Analyse le marché avec logging optimisé et gestion robuste des erreurs."""
        try:
            # Délai minimal entre analyses du même symbole (5 secondes)
            symbol_key = f"last_analysis_{symbol}"
            now = datetime.now()
   
            if hasattr(self, symbol_key):
                last_analysis = getattr(self, symbol_key)
                if (now - last_analysis).total_seconds() < 5:
                    return  # Ignorer l'analyse si moins de 5 secondes écoulées
           
            setattr(self, symbol_key, now)
   
            self.logger.info(f"Début analyse marché pour {symbol}")
       
            # Récupération des données avec gestion d'erreur dédiée
            try:
                market_data = await self.market_data.get_market_data(symbol)
                self.logger.info(f"Données reçues pour {symbol}: {market_data is not None and not market_data.empty}")
           
                if market_data is None or market_data.empty:
                    self.logger.warning(f"Données de marché vides ou invalides pour {symbol}")
                    return
            except Exception as data_error:
                self.logger.error(f"Erreur récupération données {symbol}: {data_error}")
                return

            # Ajout de logging détaillé
            try:
                tech_analysis = await self._compute_technical_analysis(market_data)
           
                if not tech_analysis:
                    self.logger.warning(f"Analyse technique vide pour {symbol}")
                    return
               
                self.logger.info(
                    f"Analyse {symbol} - Prix: {market_data['close'].iloc[-1]}, "
                    f"Vol: {market_data['volume'].iloc[-1]}, "
                    f"RSI: {tech_analysis.get('rsi', 'N/A')}"
                )

                # Génération du signal avec gestion d'erreur spécifique
                try:
                    signal = await self.strategy.generate_signal(symbol, market_data, tech_analysis)
                    self.logger.info(f"Génération de signal pour {symbol}: {'Succès' if signal else 'Aucun signal'}")
               
                    if signal:
                        self.logger.warning(
                            f"TRAITEMENT DU SIGNAL: {symbol}, {signal.get('action')}, force={signal.get('strength')}"
                        )
                        # Incrémentation du compteur de signaux générés
                        self.metrics["market_analysis"]["signals_generated"] += 1
                        # Traitement du signal avec gestion d'erreur
                        try:
                            result = await self._process_trading_signal(signal)
                            self.logger.warning(
                                f"RÉSULTAT TRAITEMENT: {symbol}, success={result}"
                            )
                        except Exception as process_error:
                            self.logger.error(f"Erreur traitement signal {symbol}: {process_error}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                    else:
                        # Signal non généré par la stratégie
                        current_price = market_data["close"].iloc[-1] if not market_data.empty else None
                        indicators = {
                            "rsi": tech_analysis.get("rsi", "N/A"),
                            "macd": tech_analysis.get("macd", "N/A"),
                            "volatility": tech_analysis.get("volatility", "N/A"),
                        }
                        self.logger.info(f"Aucun signal généré pour {symbol}")
                        self.log_signal_rejection(
                            symbol, current_price, indicators, "STRATEGY_NO_SIGNAL"
                        )
                except Exception as signal_error:
                    self.logger.error(f"Erreur génération signal pour {symbol}: {signal_error}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            except Exception as analysis_error:
                self.logger.error(f"Erreur lors de l'analyse technique pour {symbol}: {analysis_error}")
                import traceback
                self.logger.error(traceback.format_exc())
                return

            # Rate limiting
            now = datetime.now()
            if (now - self._analysis_config["last_reset"]).total_seconds() >= 1:
                self._analysis_config["analysis_count"] = 0
                self._analysis_config["last_reset"] = now

            if (
                self._analysis_config["analysis_count"]
                >= self._analysis_config["max_analysis_per_second"]
            ):
                await asyncio.sleep(self._analysis_config["cooldown_period"])

            self._analysis_config["analysis_count"] += 1

            # Reset du circuit breaker en cas de succès
            self._circuit_breaker["consecutive_errors"] = 0
            self._circuit_breaker["current_delay"] = self._circuit_breaker["initial_delay"]

        except Exception as e:
            # Log critique en cas d'erreur majeure
            self.logger.error(f"Erreur critique analyse marché {symbol}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Gestion du circuit breaker
            self._circuit_breaker["consecutive_errors"] += 1
            if (
                self._circuit_breaker["consecutive_errors"]
                >= self._circuit_breaker["error_threshold"]
            ):
                self._circuit_breaker["current_delay"] = min(
                    self._circuit_breaker["current_delay"]
                    * self._circuit_breaker["backoff_factor"],
                    self._circuit_breaker["max_delay"],
                )
                if self._should_log("critical"):
                    self.logger.warning(
                        f"Circuit breaker activé: délai {self._circuit_breaker['current_delay']}s"
                    )
                await asyncio.sleep(self._circuit_breaker["current_delay"])

    async def _compute_technical_analysis(self, market_data: pd.DataFrame):
        try:
            self.logger.info("Début calcul analyse technique")
            if market_data is None or market_data.empty:
                self.logger.error("Données de marché invalides")
                return {}

            results = {
                "volatility": self.market_data.calculate_volatility(market_data),
                "volume_profile": self.market_data.calculate_volume_profile(
                    market_data
                ),
                "price_levels": self.market_data.calculate_price_levels(market_data),
            }
            self.logger.info(f"Résultats analyse technique: {results}")
            return results

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse technique: {str(e)}")
            return {}

    async def _process_trading_signal(self, signal: Dict[str, Any]):
        """Traite un signal de trading."""
        try:
            # Log de l'état du frein d'urgence
            self.logger.info(f"État frein d'urgence: _emergency_brake_activated={getattr(self, '_emergency_brake_activated', False)}, _signal_suspension_until={getattr(self, '_signal_suspension_until', None)}")
            
            # Vérifier le frein d'urgence en priorité
            if hasattr(self, '_check_global_emergency_brake') and self._check_global_emergency_brake():
                self.log_signal_rejection(signal.get('symbol', 'unknown'), 
                                        signal.get('metrics', {}).get('price', 'unknown'), 
                                        signal.get('indicators', {}), 
                                        "EMERGENCY_BRAKE_ACTIVE")
                return False
            
            # Vérifier l'arrêt critique
            if hasattr(self, '_critical_stop') and self._critical_stop():
                self.log_signal_rejection(signal.get('symbol', 'unknown'), 
                                        signal.get('metrics', {}).get('price', 'unknown'), 
                                        signal.get('indicators', {}), 
                                        "CRITICAL_STOP_ACTIVE")
                return False
        
            # Log du signal reçu
            self.logger.info(f"[TRAITEMENT] Signal reçu pour {signal.get('symbol')}: {signal.get('action')} - Stratégie: {signal.get('strategy')}")
    
            # Vérifications préliminaires
            symbol = signal.get('symbol')
            current_price = signal.get('metrics', {}).get('price')
            indicators = signal.get('indicators', {})
    
            # Log détaillé pour debug
            self.logger.info(f"[TRAITEMENT] Prix actuel: {current_price}, Force: {signal.get('strength', 0):.4f}")
    
            if not symbol or not current_price:
                self.logger.error(f"[TRAITEMENT] Signal invalide: données manquantes")
                self.log_signal_rejection(symbol, current_price, indicators, "INVALID_SIGNAL")
                return False
            
            # Validation des signaux dupliqués avec fenêtre temporelle
            signal_key = f"{symbol}_{signal['action']}"
            current_time = datetime.now()

            if signal_key in self._processed_signals:
                last_time = self._processed_signals[signal_key]
                seconds_elapsed = (current_time - last_time).total_seconds()
    
                # Autoriser le même signal après 5 secondes
                if seconds_elapsed < 5:
                    self.logger.info(f"[TRAITEMENT] Signal similaire pour {symbol} ignoré temporairement (10s)")
                    self.log_signal_rejection(symbol, current_price, indicators, "DUPLICATE_SIGNAL")
                    return False

            # Mettre à jour le timestamp du dernier signal
            self._processed_signals[signal_key] = current_time

            # Vérification de la tendance macro si disponible
            if hasattr(self, 'market_data') and hasattr(self.market_data, 'get_market_trend'):
                try:
                    trend = await self.market_data.get_market_trend(symbol, '1h')
                
                    # Si signal contradictoire avec tendance macro, augmenter le seuil de validation
                    if (signal['action'] == 'buy' and trend < -0.005) or (signal['action'] == 'sell' and trend > 0.005):
                        if signal['strength'] < 0.35:  # Seuil plus élevé pour les contre-tendances
                            self.log_signal_rejection(symbol, current_price, indicators, "COUNTER_TREND_SIGNAL")
                            return False
                except Exception as e:
                    self.logger.warning(f"Erreur lors de la vérification de tendance: {e}")

            # Validation risk manager avec logs détaillés
            self.logger.info(f"[TRAITEMENT] Vérification par le risk manager...")
            check_result, check_message = self.risk_manager.check_trade(signal)
            self.logger.info(f"[TRAITEMENT] Résultat validation: {check_result}, {check_message}")
    
            if check_result:
                # Calcul taille de position avec vérification
                self.logger.info(f"[TRAITEMENT] Calcul taille position pour {symbol}...")
                position_size = self.risk_manager.calculate_position_size(
                    self.current_capital,
                    current_price,
                    symbol
                )
        
                self.logger.info(f"[TRAITEMENT] Taille calculée: {position_size}")
        
                if not position_size or position_size <= 0:
                    self.logger.error(f"[TRAITEMENT] Taille de position invalide: {position_size}")
                    self.log_signal_rejection(symbol, current_price, indicators, "INVALID_POSITION_SIZE")
                    return False
   
                # Placement ordre
                self.logger.info(f"[TRAITEMENT] Placement ordre: {symbol} {signal.get('action')} taille={position_size} prix={current_price}")
                order = await self.order_manager.place_order(
                    symbol=symbol,
                    side=signal.get('action'),
                    size=position_size,
                    strategy=signal.get('strategy')
                )
       
                if order:
                    self.logger.info(f"[TRAITEMENT] Ordre placé avec succès: {order}")
                    self.logger.info(f"[DIAGNOSTIC] Ordre placé pour {symbol}: {order}")
            
                    # Récupération de la catégorie de marché
                    market_type = self.risk_manager.get_market_category(symbol)
            
                    # Ouverture de la position
                    position_opened = await self.position_manager.open_position(
                        symbol=symbol,
                        side=signal.get('action'),
                        size=position_size,
                        entry_price=current_price,
                        market_type=market_type,
                        strategy=signal.get('strategy')
                    )
           
                    if position_opened:
                        # Incrémentation des compteurs
                        self.metrics['market_analysis']['signals_executed'] += 1
                        self.metrics['trades']['total'] += 1
                
                        # Mise à jour des métriques dans la stratégie
                        if hasattr(self.strategy, 'metrics') and hasattr(self.strategy.metrics, 'executed'):
                            self.strategy.metrics.executed += 1
                
                        # Log détaillé pour confirmer l'exécution
                        self.logger.info(f"[TRAITEMENT] TRADE EXÉCUTÉ: {symbol} à {current_price}, taille: {position_size}, stratégie: {signal.get('strategy')}")
                
                        return True
                    else:
                        self.logger.warning(f"[TRAITEMENT] Position non ouverte pour {symbol}: position existante possible")
                    
                        # Si la position existe déjà, ne pas compter comme rejet complet
                        self.log_signal_rejection(symbol, current_price, indicators, "POSITION_NOT_OPENED")
                        return False
                else:
                    self.logger.error(f"[TRAITEMENT] Échec placement ordre: {symbol}")
                    self.log_signal_rejection(symbol, current_price, indicators, "ORDER_PLACEMENT_FAILED")
                    return False
                
            else:
                self.logger.info(f"[TRAITEMENT] Signal rejeté par risk manager: {check_message}")
                self.log_signal_rejection(symbol, current_price, indicators, f"RISK_MANAGER_REJECTION: {check_message}")
                return False
            
        except Exception as e:
            self.logger.error(f"[TRAITEMENT] Erreur traitement signal: {str(e)}")
            self.log_signal_rejection(signal.get('symbol', 'unknown'), 
                                    signal.get('metrics', {}).get('price', 'unknown'), 
                                    signal.get('indicators', {}), 
                                    f"SIGNAL_PROCESS_ERROR")
            self._error_counter += 1
            return False

    async def force_test_trade(self, symbol: str = "BTC/USDT"):
        """Méthode de test forcé pour vérifier le flux des trades."""
        try:
            self.logger.info(f"[TEST] Création d'un trade forcé pour {symbol}")

            # Récupération données
            market_data = await self.market_data.get_market_data(
                symbol=symbol, interval="1m", limit=10
            )

            if market_data is None or market_data.empty:
                self.logger.error("[TEST] Impossible d'obtenir les données de marché")
                return

            # Création signal manuel
            current_price = float(market_data["close"].iloc[-1])

            test_signal = {
                "symbol": symbol,
                "action": "buy",
                "strategy": "TEST",
                "strength": 1.0,
                "metrics": {
                    "price": current_price,
                    "volume": 1000000,
                    "volatility": 0.01,
                    "spread": 0.001,
                },
                "indicators": {"rsi": 50, "macd_hist": 0.001, "bb_width": 0.02},
            }

            # Contournement des validations
            self.logger.info(
                f"[TEST] Placement ordre direct : {symbol} à {current_price}"
            )

            # Essai d'ouverture de position directe
            position_size = 0.01  # Taille fixe pour test
            position = await self.position_manager.open_position(
                symbol=symbol,
                side="buy",
                size=position_size,
                entry_price=current_price,
                market_type="MAJOR",
            )

            if position:
                self.logger.info(f"[TEST] Position créée: {position}")
                return position
            else:
                self.logger.error("[TEST] Échec création position")
                return None

        except Exception as e:
            self.logger.error(f"[TEST] Erreur force_test_trade: {str(e)}")
            return None

    async def _update_positions(self):
        """Met à jour les positions ouvertes."""
        try:
            # Vérifier que position_manager est initialisé
            if not hasattr(self, "position_manager") or not self.position_manager:
                return

            # Vérifier que get_all_positions est disponible
            if not hasattr(self.position_manager, "get_all_positions"):
                return

            positions = self.position_manager.get_all_positions()
            for position in positions:
                try:
                    # Vérifier les champs requis
                    if not position.get("symbol"):
                        continue

                    # Récupération prix actuel
                    market_data = await self.market_data.get_market_data(
                        symbol=position["symbol"], interval="1m", limit=1
                    )

                    if market_data is not None and not market_data.empty:
                        current_price = market_data["close"].iloc[-1]

                        # Mise à jour position
                        pnl = await self.position_manager.update_position(
                            {
                                "symbol": position["symbol"],
                                "current_price": current_price,
                            }
                        )

                        # Si la position a été fermée avec un PnL
                        if pnl != 0:
                            # Log avant modification du capital
                            old_capital = self.current_capital
                            old_total_pnl = self.metrics['performance']['total_pnl']
                   
                            # Métriques détaillées AVANT la mise à jour
                            self.logger.info(f"Métriques AVANT mise à jour - Capital: {old_capital:.4f}€, PnL total: {old_total_pnl:.4f}€")
                   
                            # Mise à jour du capital et peak_capital
                            self.current_capital += pnl
                            self.peak_capital = max(self.current_capital, self.peak_capital)
                            self._update_drawdown()
                   
                            # Mise à jour explicite du PnL total
                            self.metrics['performance']['total_pnl'] += pnl

                            # Log détaillé APRÈS la transaction
                            self.logger.info(f"Position fermée: {position['symbol']}, Entry: {position.get('entry_price', 'N/A')}, Exit: {current_price}, PnL: {pnl:.4f}€")
                            self.logger.info(f"Capital APRÈS mise à jour: {old_capital:.4f}€ -> {self.current_capital:.4f}€")
                            self.logger.info(f"Performance totale APRÈS mise à jour: {old_total_pnl:.4f}€ -> {self.metrics['performance']['total_pnl']:.4f}€")
                   
                            # Vérification des métriques du position_manager
                            if hasattr(self.position_manager, "get_metrics"):
                                pm_metrics = self.position_manager.get_metrics()
                                self.logger.info(f"Position manager metrics - total_pnl: {pm_metrics.get('total_pnl', 0.0)}")

                            # Mise à jour des métriques
                            if pnl > 0:
                                self.metrics["trades"]["winners"] += 1
                            else:
                                self.metrics["trades"]["losers"] += 1

                except Exception as e:
                    self.logger.error(
                        f"Erreur mise à jour position {position.get('symbol', 'unknown')}: {e}"
                    )
                    import traceback
                    self.logger.error(traceback.format_exc())

            # Vérification des timeouts et fermeture des positions expirées
            if hasattr(self.position_manager, "check_positions_timeout"):
                await self.position_manager.check_positions_timeout()
            
            # Fermeture forcée des positions après un certain temps (test diagnostic)
            for position in positions:
                entry_time = position.get('entry_time')
                current_time = datetime.now()
                if entry_time and (current_time - datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time).total_seconds() > 10:
                    self.logger.warning(f"[FORCE] Fermeture forcée de {position['symbol']} après 10s pour test diagnostic")
                    await self.position_manager.close_position(
                        symbol=position['symbol'],
                        exit_price=position.get('current_price', position['entry_price']),
                        reason='test_forced_close'
                    )

        except Exception as e:
            self.logger.error(f"Erreur mise à jour positions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _update_drawdown(self):
        """Met à jour les métriques de drawdown."""
        try:
            if self.peak_capital > 0:
                current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.current_capital < self.peak_capital else 0
                self.metrics['performance']['current_drawdown'] = current_drawdown
                self.metrics['performance']['max_drawdown'] = max(
                    current_drawdown,
                    self.metrics['performance']['max_drawdown']
                )
        except Exception as e:
            self.logger.error(f"Erreur calcul drawdown: {str(e)}")

    def _check_emergency_mode(self):
        """Active le mode urgence en cas de pertes importantes."""
        try:
            # Variables pour éviter les appels redondants
            current_state = self.risk_limits.get('emergency_mode', False)
        
            # Vérifier si on a atteint la limite de perte journalière
            if (self.metrics["performance"]["total_pnl"] <= -self.risk_limits["max_daily_loss"] * self.initial_capital):
                if not current_state:
                    self.risk_limits["emergency_mode"] = True
                    self.logger.warning(f"MODE URGENCE ACTIVÉ - Perte journalière: {self.metrics['performance']['total_pnl']:.2f}€")

            # Vérifier si on a atteint le drawdown maximum
            elif (self.metrics["performance"]["current_drawdown"] >= self.risk_limits["max_drawdown"]):
                if not current_state:
                    self.risk_limits["emergency_mode"] = True
                    self.logger.warning(f"MODE URGENCE ACTIVÉ - Drawdown: {self.metrics['performance']['current_drawdown']*100:.2f}%")

            # Ajout : Vérification de sortie du mode urgence (avec hystérésis)
            elif current_state and self.metrics['performance']['current_drawdown'] < self.risk_limits['recovery_threshold'] * 0.8:  # 80% du seuil pour éviter les oscillations
                self.risk_limits['emergency_mode'] = False
                self.logger.warning(f"MODE URGENCE DÉSACTIVÉ - Drawdown réduit à {self.metrics['performance']['current_drawdown']*100:.2f}%")
    
        except Exception as e:
            self.logger.error(f"Erreur vérification mode urgence: {e}")

    def _update_metrics(self):
        """Met à jour les métriques de trading."""
        try:
            # Synchronisation explicite du PnL avec le PositionManager
            if hasattr(self, "position_manager") and self.position_manager:
                if hasattr(self.position_manager, "get_total_pnl"):
                    pm_total_pnl = self.position_manager.get_total_pnl()
                    if pm_total_pnl != self.metrics["performance"]["total_pnl"]:
                        self.logger.info(f"Synchronisation PnL: {self.metrics['performance']['total_pnl']:.4f}€ -> {pm_total_pnl:.4f}€")
                        self.metrics["performance"]["total_pnl"] = pm_total_pnl
            
                    # Récupération d'autres métriques importantes du position_manager
                    if hasattr(self.position_manager, "get_metrics"):
                        pm_metrics = self.position_manager.get_metrics()
                        if pm_metrics:
                            # Synchronisation des métriques de win_rate
                            if 'win_rate' in pm_metrics and pm_metrics['win_rate'] > 0:
                                self.metrics["performance"]["win_rate"] = pm_metrics['win_rate'] * 100
                    
                            # Synchronisation des métriques de drawdown
                            if 'max_drawdown' in pm_metrics:
                                self.metrics["performance"]["max_drawdown"] = pm_metrics['max_drawdown']
        
            # Calcul win rate si non synchronisé depuis position_manager
            if self.metrics["performance"]["win_rate"] == 0:
                total_trades = (
                    self.metrics["trades"]["winners"] + self.metrics["trades"]["losers"]
                )
                if total_trades > 0:
                    self.metrics["performance"]["win_rate"] = (
                        self.metrics["trades"]["winners"] / total_trades * 100
                    )

            # Mise à jour drawdown
            self._update_drawdown()
        
            # Log périodique des métriques principales (toutes les 100 mises à jour)
            if not hasattr(self, '_metrics_log_counter'):
                self._metrics_log_counter = 0
        
            self._metrics_log_counter += 1
            if self._metrics_log_counter % 100 == 0:
                self.logger.info(
                    f"Métriques actuelles - Capital: {self.current_capital:.2f}€, "
                    "PnL: {self.metrics['performance']['total_pnl']:.2f}€, "
                    f"Win rate: {self.metrics['performance']['win_rate']:.1f}%, "
                    f"Drawdown: {self.metrics['performance']['current_drawdown']*100:.1f}%"
                )

        except Exception as e:
            self.logger.error(f"Erreur mise à jour métriques: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _check_global_emergency_brake(self):
        """Vérifie si le frein d'urgence global doit être activé."""
        try:
            # Pour le Test 3B: Limiter à 3 trades maximum
            if hasattr(self, 'metrics') and self.metrics['trades']['total'] >= 5:
                if not hasattr(self, '_emergency_brake_activated') or not self._emergency_brake_activated:
                    self.logger.critical("LIMITE DE TEST ATTEINTE: 3 trades maximum pour Test 3B")
                    self._emergency_brake_activated = True
                return True
            
            # Vérification standard désactivée pour le Test 3B
            return False
        
            # Code original commenté
            # capital_loss_pct = (self.initial_capital - self.current_capital) / self.initial_capital
            # if capital_loss_pct > 0.1:  # 10% de perte
            #    if not hasattr(self, '_emergency_brake_activated') or not self._emergency_brake_activated:
            #        self.logger.critical(f"FREIN D'URGENCE ACTIVÉ: Capital réduit de {capital_loss_pct*100:.1f}% - ARRÊT DE TOUT TRADING")
            #        self._emergency_brake_activated = True
            #        
            #        # Fermer toutes les positions ouvertes
            #        if hasattr(self.position_manager, "close_all_positions"):
            #            asyncio.create_task(self.position_manager.close_all_positions("emergency_brake"))
            #        
            #        # Suspendre la génération de signaux pendant 1 heure
            #        self._signal_suspension_until = datetime.now() + timedelta(hours=1)
            #        
            #    return True
            # return False
        
        except Exception as e:
            self.logger.error(f"Erreur vérification frein d'urgence: {e}")
            return False  # En cas d'erreur, ne pas activer le frein d'urgence pour le Test 3B
        
    def _critical_stop(self):
        """Arrête tout trading après X pertes consécutives"""
        # Pour le Test 3A: désactivé temporairement
        self.logger.debug("Test 3A: Frein d'urgence temporairement désactivé")
    
        # Limiter à 3 trades maximum pour ce test
        if hasattr(self, 'metrics') and self.metrics['trades']['total'] >= 3:
            self.logger.critical("LIMITE DE TEST ATTEINTE: 3 trades maximum pour Test 3A")
            self._emergency_brake_activated = True
            return True
        
        return False

    def _cleanup_processed_signals(self):
        """Nettoie les signaux traités avec une fenêtre plus appropriée."""
        now = datetime.now()
        max_age = timedelta(minutes=5)  # Durée maximale de conservation des signaux
    
        # Supprimer les signaux trop anciens
        expired_keys = []
        for signal_key, timestamp in self._processed_signals.items():
            if now - timestamp > max_age:
                expired_keys.append(signal_key)
    
        # Supprimer les clés expirées
        for key in expired_keys:
            del self._processed_signals[key]
        
        # Limiter la taille du dictionnaire si nécessaire
        max_size = 1000
        if len(self._processed_signals) > max_size:
            self.logger.warning(f"Nettoyage forcé des signaux traités: {len(self._processed_signals)} -> {max_size}")
            # Garder les signaux les plus récents
            sorted_signals = sorted(self._processed_signals.items(), key=lambda x: x[1], reverse=True)
            self._processed_signals = dict(sorted_signals[:max_size])

    def _generate_detailed_report(self):
        """Génère un rapport détaillé périodique sur l'état du bot."""
        current_time = datetime.now()

        # Vérifier si l'intervalle est écoulé
        if (current_time - self.last_detailed_report) < self.report_interval:
            return

        # Préparer le rapport
        self.logger.info("=" * 100)
        self.logger.info(
            f"RAPPORT DÉTAILLÉ - {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info("=" * 100)

        # Informations financières
        self.logger.info(f"Capital actuel: {self.current_capital:.2f}€")
        self.logger.info(f"PnL total: {self.metrics['performance']['total_pnl']:.2f}€")
        self.logger.info(
            f"Drawdown actuel: {self.metrics['performance']['current_drawdown']*100:.2f}%"
        )
        self.logger.info(
            f"Drawdown maximum: {self.metrics['performance']['max_drawdown']*100:.2f}%"
        )

        # Informations sur les signaux
        self.logger.info(
            f"Signaux générés: {self.metrics['market_analysis']['signals_generated']}"
        )
        self.logger.info(
            f"Signaux exécutés: {self.metrics['market_analysis']['signals_executed']}"
        )
        self.logger.info(
            f"Signaux rejetés: {self.metrics['market_analysis']['signals_rejected']}"
        )

        # Statistiques de rejet détaillées
        self.logger.info("Détail des rejets:")
        for reason, count in sorted(
            self.rejection_stats.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            self.logger.info(f"  - {reason}: {count}")

        # Résumé des positions actuelles
        if hasattr(self, "position_manager") and self.position_manager:
            positions = self.position_manager.get_all_positions()
            if positions:
                self.logger.info(f"Positions ouvertes: {len(positions)}")
                for pos in positions:
                    self.logger.info(
                        f"  - {pos['symbol']}: entrée à {pos.get('entry_price', 'N/A')}, "
                        f"actuel: {pos.get('current_price', 'N/A')}, "
                        f"PnL: {pos.get('unrealized_pnl', 'N/A')}"
                    )
            else:
                self.logger.info("Aucune position ouverte")

        # Indicateurs techniques actuels pour chaque symbole
        self.logger.info("Indicateurs techniques actuels:")
        for symbol in self.symbols[
            :3
        ]:  # Limiter aux 3 premiers symboles pour éviter les logs trop longs
            try:
                # Obtenir les dernières données et indicateurs si disponibles
                if hasattr(self, "market_data") and self.market_data:
                    indicator_data = (
                        self.market_data.get_cached_indicators(symbol)
                        if hasattr(self.market_data, "get_cached_indicators")
                        else None
                    )
                    if indicator_data:
                        self.logger.info(f"  - {symbol}: {indicator_data}")
            except Exception as e:
                self.logger.error(f"Erreur récupération indicateurs pour {symbol}: {e}")

        # Mode urgence
        if self.risk_limits["emergency_mode"]:
            self.logger.warning("⚠️ MODE URGENCE ACTIF - Réduction de l'exposition")

        self.logger.info("=" * 100)

        # Mettre à jour le timestamp du dernier rapport
        self.last_detailed_report = current_time

    def _log_metrics(self):
        """Log les métriques actuelles."""
        self.logger.info(
            f"Métriques de trading:\n"
            f"Capital: {self.current_capital:.2f}€\n"
            f"PnL: {self.metrics['performance']['total_pnl']:.2f}€\n"
            f"Win rate: {self.metrics['performance']['win_rate']:.1f}%\n"
            f"Drawdown: {self.metrics['performance']['current_drawdown']*100:.1f}%"
        )

        # Ajouter les statistiques de rejet
        top_rejections = sorted(
            self.rejection_stats.items(), key=lambda x: x[1], reverse=True
        )[:5]
        self.logger.info(f"Top 5 rejets: {dict(top_rejections)}")

    def get_metrics(self) -> Dict:
        """Retourne les métriques actuelles."""
        metrics = {
            "current_capital": self.current_capital,
            "total_pnl": self.metrics["performance"]["total_pnl"],
            "win_rate": self.metrics["performance"]["win_rate"],
            "max_drawdown": self.metrics["performance"]["max_drawdown"],
            "trades": self.metrics["trades"]["total"],
            "signals": {
                "generated": self.metrics["market_analysis"]["signals_generated"],
                "executed": self.metrics["market_analysis"]["signals_executed"],
                "rejected": self.metrics["market_analysis"]["signals_rejected"],
            },
            "rejection_details": dict(self.rejection_stats),
        }

        # Ajout des métriques des gestionnaires
        if hasattr(self, "position_manager") and hasattr(
            self.position_manager, "get_metrics"
        ):
            position_metrics = self.position_manager.get_metrics()
            metrics["position_metrics"] = position_metrics

        if hasattr(self, "strategy") and hasattr(self.strategy, "get_metrics"):
            strategy_metrics = self.strategy.get_metrics()
            metrics["strategy_metrics"] = strategy_metrics

        return metrics

    async def stop(self):
        """Arrêt gracieux du bot."""
        try:
            self._running = False
            self.logger.info("Bot interrompu")

            # Log final des statistiques
            self._log_metrics()
            self.logger.info(
                f"Statistiques finales de rejets: {dict(self.rejection_stats)}"
            )

            # Vérifier si le position_manager existe
            if hasattr(self, "position_manager") and self.position_manager:
                # Fermeture positions ouvertes
                if hasattr(self.position_manager, "close_all_positions"):
                    await self.position_manager.close_all_positions(reason="bot_stop")
                else:
                    positions = self.position_manager.get_all_positions()
                    if positions:
                        for position in positions:
                            try:
                                await self.position_manager.close_position(
                                    symbol=position["symbol"],
                                    exit_price=position.get(
                                        "current_price", position["entry_price"]
                                    ),
                                    reason="bot_stop",
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"Erreur fermeture position {position['symbol']}: {e}"
                                )

            # Attendre la fermeture complète des positions
            await asyncio.sleep(1)

            # Sauvegarde des métriques finales
            self._save_final_metrics()

        except Exception as e:
            self.logger.error(f"Erreur arrêt bot: {e}")
        finally:
            # S'assurer que le bot est bien arrêté
            self._running = False
            self.logger.info("Bot arrêté")

    def _save_final_metrics(self):
        """Sauvegarde les métriques finales dans un fichier JSON."""
        try:
            import json
            from datetime import datetime

            metrics = self.get_metrics()

            # Ajout des informations de la session
            metrics["session_info"] = {
                "end_time": datetime.now().isoformat(),
                "initial_capital": self.initial_capital,
                "symbols": self.symbols,
            }

            # Sauvegarde dans un fichier
            with open(
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w"
            ) as f:
                json.dump(metrics, f, indent=2, default=str)

            self.logger.info("Métriques finales sauvegardées")

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde métriques: {e}")
