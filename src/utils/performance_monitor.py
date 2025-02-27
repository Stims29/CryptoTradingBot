import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

class PerformanceMonitor:
    def __init__(self):
        """Initialise le moniteur de performance"""
        self.metrics = {
            'signals': [],
            'trades': [],
            'daily_performance': {}
        }
        
        # Création du dossier metrics s'il n'existe pas
        self.metrics_dir = Path('results/metrics')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def log_signal(self, symbol: str, signal: Dict):
        """
        Enregistre un signal de trading
        
        Args:
            symbol: Symbole concerné
            signal: Signal généré par la stratégie
        """
        try:
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'price': signal.get('price', 0),
                'components_used': signal.get('components_used', {})
            }
            
            self.metrics['signals'].append(signal_data)
            self._save_metrics('signals')
            
        except Exception as e:
            logging.error(f"Erreur lors de l'enregistrement du signal: {str(e)}")

    def log_trade(self, trade_data: Dict):
        """
        Enregistre un trade
        
        Args:
            trade_data: Données du trade
        """
        try:
            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_data['symbol'],
                'type': trade_data['type'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data.get('exit_price'),
                'size': trade_data['size'],
                'pnl': trade_data.get('pnl', 0),
                'pnl_percent': trade_data.get('pnl_percent', 0)
            }
            
            self.metrics['trades'].append(trade_info)
            self._save_metrics('trades')
            
            # Mise à jour des performances quotidiennes
            self._update_daily_performance(trade_info)
            
        except Exception as e:
            logging.error(f"Erreur lors de l'enregistrement du trade: {str(e)}")

    def _update_daily_performance(self, trade_info: Dict):
        """
        Met à jour les métriques de performance quotidiennes
        
        Args:
            trade_info: Informations sur le trade
        """
        try:
            date = datetime.now().strftime('%Y-%m-%d')
            
            if date not in self.metrics['daily_performance']:
                self.metrics['daily_performance'][date] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0,
                    'max_profit': 0,
                    'max_loss': 0
                }
                
            daily_stats = self.metrics['daily_performance'][date]
            pnl = trade_info.get('pnl', 0)
            
            daily_stats['total_trades'] += 1
            daily_stats['total_pnl'] += pnl
            
            if pnl > 0:
                daily_stats['winning_trades'] += 1
                daily_stats['max_profit'] = max(daily_stats['max_profit'], pnl)
            elif pnl < 0:
                daily_stats['losing_trades'] += 1
                daily_stats['max_loss'] = min(daily_stats['max_loss'], pnl)
                
            # Calcul des métriques additionnelles
            daily_stats['win_rate'] = (daily_stats['winning_trades'] / daily_stats['total_trades']) * 100
            
            self._save_metrics('daily_performance')
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des performances quotidiennes: {str(e)}")

    def get_performance_summary(self) -> Dict:
        """
        Génère un résumé des performances
        
        Returns:
            Dict contenant les métriques de performance
        """
        try:
            total_trades = len(self.metrics['trades'])
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_trade_pnl': 0
                }
            
            winning_trades = sum(1 for trade in self.metrics['trades'] if trade.get('pnl', 0) > 0)
            total_pnl = sum(trade.get('pnl', 0) for trade in self.metrics['trades'])
            
            return {
                'total_trades': total_trades,
                'win_rate': (winning_trades / total_trades) * 100,
                'total_pnl': total_pnl,
                'avg_trade_pnl': total_pnl / total_trades
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération du résumé des performances: {str(e)}")
            return {}

    def _save_metrics(self, metric_type: str):
        """
        Sauvegarde les métriques dans un fichier
        
        Args:
            metric_type: Type de métrique à sauvegarder
        """
        try:
            filename = self.metrics_dir / f'{metric_type}_{datetime.now().strftime("%Y%m%d")}.json'
            
            with open(filename, 'w') as f:
                json.dump(self.metrics[metric_type], f, indent=4)
                
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des métriques: {str(e)}")

    def export_metrics(self, format: str = 'csv'):
        """
        Exporte les métriques dans le format spécifié
        
        Args:
            format: Format d'export ('csv' ou 'json')
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == 'csv':
                # Export des signaux
                pd.DataFrame(self.metrics['signals']).to_csv(
                    self.metrics_dir / f'signals_{timestamp}.csv',
                    index=False
                )
                
                # Export des trades
                pd.DataFrame(self.metrics['trades']).to_csv(
                    self.metrics_dir / f'trades_{timestamp}.csv',
                    index=False
                )
                
                # Export des performances quotidiennes
                pd.DataFrame.from_dict(
                    self.metrics['daily_performance'],
                    orient='index'
                ).to_csv(self.metrics_dir / f'daily_performance_{timestamp}.csv')
                
            elif format == 'json':
                with open(self.metrics_dir / f'metrics_{timestamp}.json', 'w') as f:
                    json.dump(self.metrics, f, indent=4)
                    
            logging.info(f"Métriques exportées au format {format}")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'export des métriques: {str(e)}")