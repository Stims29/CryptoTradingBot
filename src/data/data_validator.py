#!/usr/bin/env python
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..exceptions import ValidationError
from utils.logging import setup_logging

class DataValidator:
    def __init__(self):
        """Initialise le validateur de données."""
        self.logger = setup_logging(__name__)
        
        # Règles de validation par type de donnée
        self.validation_rules = {
            'OHLCV': {
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'min_rows': 10,           # Réduit de 20 à 10 pour plus de réactivité
                'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
                'positive_columns': ['volume'],
                'ohlc_rules': [
                    'high >= max(open, close)',
                    'low <= min(open, close)'
                ],
                'max_gap': 5,            # Maximum 5 secondes entre points
                'min_volume': 0.1,       # Volume minimum 0.1
                'max_spread': 20         # Spread maximum 20%
            },
            'TRADE': {
                'required_fields': ['symbol', 'side', 'size', 'price', 'timestamp'],
                'numeric_fields': ['size', 'price'],
                'positive_fields': ['size', 'price'],
                'min_size': 0.1,         # Taille minimum 0.1
                'max_size': 5.0          # Taille maximum 5.0
            },
            'ORDER': {
                'required_fields': ['symbol', 'side', 'type', 'size'],
                'numeric_fields': ['size', 'price'],
                'positive_fields': ['size'],
                'valid_sides': ['buy', 'sell'],
                'valid_types': ['market', 'limit']
            }
        }

    def validate_ohlcv(self, data: pd.DataFrame, symbol: str) -> bool:
        """Valide les données OHLCV."""
        try:
            rules = self.validation_rules['OHLCV']
            
            # Vérification colonnes requises
            missing_cols = set(rules['required_columns']) - set(data.columns)
            if missing_cols:
                self.logger.warning(
                    f"{symbol}: Colonnes manquantes: {missing_cols}\n"
                    f"Colonnes présentes: {list(data.columns)}"
                )
                return False
            
            # Vérification nombre de lignes
            if len(data) < rules['min_rows']:
                self.logger.warning(
                    f"{symbol}: Pas assez de données ({len(data)} < {rules['min_rows']})"
                )
                return False
            
            # Vérification valeurs numériques
            for col in rules['numeric_columns']:
                if not pd.to_numeric(data[col], errors='coerce').notnull().all():
                    self.logger.warning(f"{symbol}: Valeurs non numériques dans {col}")
                    return False
            
            # Vérification valeurs positives
            for col in rules['positive_columns']:
                if (data[col] <= rules['min_volume']).any():
                    self.logger.warning(
                        f"{symbol}: Volume insuffisant (min: {rules['min_volume']})"
                    )
                    return False
            
            # Vérification règles OHLC
            if not (data['high'] >= data[['open', 'close']].max(axis=1)).all():
                self.logger.warning(f"{symbol}: High doit être >= max(open, close)")
                return False
                
            if not (data['low'] <= data[['open', 'close']].min(axis=1)).all():
                self.logger.warning(f"{symbol}: Low doit être <= min(open, close)")
                return False
            
            # Vérification gaps temporels
            if data.index.dtype == 'datetime64[ns]':
                gaps = data.index.to_series().diff().dt.total_seconds()
                if (gaps > rules['max_gap']).any():
                    self.logger.warning(
                        f"{symbol}: Gaps temporels > {rules['max_gap']}s détectés"
                    )
                    return False
            
            # Vérification spread
            spread_pct = ((data['high'] - data['low']) / data['low'] * 100)
            if (spread_pct > rules['max_spread']).any():
                self.logger.warning(
                    f"{symbol}: Spread excessif > {rules['max_spread']}% détecté"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation OHLCV pour {symbol}: {e}")
            return False

    def validate_trade(self, trade: Dict, strict: bool = True) -> bool:
        """Valide les données d'un trade."""
        try:
            rules = self.validation_rules['TRADE']
            
            # Vérification champs requis
            if strict:
                missing_fields = set(rules['required_fields']) - set(trade.keys())
                if missing_fields:
                    self.logger.warning(
                        f"Trade: Champs manquants: {missing_fields}\n"
                        f"Champs présents: {list(trade.keys())}"
                    )
                    return False
            
            # Vérification valeurs numériques
            for field in rules['numeric_fields']:
                if field in trade:
                    try:
                        value = float(trade[field])
                        # Vérification taille
                        if field == 'size':
                            if not rules['min_size'] <= value <= rules['max_size']:
                                self.logger.warning(
                                    f"Trade: Taille invalide: {value} "
                                    f"(min: {rules['min_size']}, max: {rules['max_size']})"
                                )
                                return False
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Trade: Valeur non numérique pour {field}: {trade[field]}"
                        )
                        return False
            
            # Vérification valeurs positives
            for field in rules['positive_fields']:
                if field in trade and float(trade[field]) <= 0:
                    self.logger.warning(
                        f"Trade: Valeur négative ou nulle pour {field}: {trade[field]}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation trade: {e}")
            return False

    def validate_order(self, order: Dict) -> bool:
        """Valide les données d'un ordre."""
        try:
            rules = self.validation_rules['ORDER']
            
            # Vérification champs requis
            missing_fields = set(rules['required_fields']) - set(order.keys())
            if missing_fields:
                self.logger.warning(
                    f"Ordre: Champs manquants: {missing_fields}\n"
                    f"Champs présents: {list(order.keys())}"
                )
                return False
            
            # Vérification type d'ordre
            if order['type'] not in rules['valid_types']:
                self.logger.warning(
                    f"Ordre: Type invalide: {order['type']}\n"
                    f"Types valides: {rules['valid_types']}"
                )
                return False
            
            # Vérification côté
            if order['side'] not in rules['valid_sides']:
                self.logger.warning(
                    f"Ordre: Côté invalide: {order['side']}\n"
                    f"Côtés valides: {rules['valid_sides']}"
                )
                return False
            
            # Vérification valeurs numériques
            for field in rules['numeric_fields']:
                if field in order:
                    try:
                        float(order[field])
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Ordre: Valeur non numérique pour {field}: {order[field]}"
                        )
                        return False
            
            # Vérification valeurs positives
            for field in rules['positive_fields']:
                if field in order and float(order[field]) <= 0:
                    self.logger.warning(
                        f"Ordre: Valeur négative ou nulle pour {field}: {order[field]}"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation ordre: {e}")
            return False

    def validate_signal(self, signal: Dict) -> bool:
        """Valide un signal de trading."""
        try:
            required_fields = [
                'symbol', 'category', 'action', 'confidence',
                'metrics', 'timestamp'
            ]
            
            # Vérification champs requis
            missing_fields = set(required_fields) - set(signal.keys())
            if missing_fields:
                self.logger.warning(
                    f"Signal: Champs manquants: {missing_fields}\n"
                    f"Champs présents: {list(signal.keys())}"
                )
                return False
            
            # Vérification action
            if signal['action'] not in ['buy', 'sell']:
                self.logger.warning(
                    f"Signal: Action invalide: {signal['action']}"
                )
                return False
            
            # Vérification confiance
            confidence = float(signal['confidence'])
            if not 0 <= confidence <= 1:
                self.logger.warning(
                    f"Signal: Confiance invalide: {confidence}"
                )
                return False
                
            # Vérification métriques
            required_metrics = ['volatility', 'volume_ratio', 'spread']
            if not all(m in signal['metrics'] for m in required_metrics):
                self.logger.warning(
                    f"Signal: Métriques manquantes\n"
                    f"Requises: {required_metrics}\n"
                    f"Présentes: {list(signal['metrics'].keys())}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur validation signal: {e}")
            return False