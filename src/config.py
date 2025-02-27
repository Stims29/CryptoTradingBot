#!/usr/bin/env python
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

class Config:
    def __init__(self, config_path: Optional[str] = None):
        """Initialise la configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Chemins par défaut
        self.base_path = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.base_path / 'config'
        
        # Configuration par défaut
        self.default_config = {
            'exchange': {
                'name': 'kucoin',
                'api_key': os.getenv('KUCOIN_API_KEY', ''),
                'api_secret': os.getenv('KUCOIN_SECRET', ''),
                'password': os.getenv('KUCOIN_PASSPHRASE', '')
            },
            'trading': {
                'simulation_mode': True,
                'initial_capital': 100.0,
                'risk_per_trade': 0.015,  # 1.5% risque par trade
                'max_trades_per_hour': 250,
                'max_exposure': 0.15,     # 15% max exposition
                'min_trade_amount': 0.5,  # 0.5€ minimum
                'max_trade_amount': 1.5   # 1.5€ maximum
            },
            'categories': {
                'MAJOR': {
                    'min_volatility': 0.01,   # 0.01% min
                    'max_volatility': 0.1,    # 0.1% max
                    'min_volume_ratio': 1.3,
                    'max_spread': 0.05,
                    'position_size': 0.015    # 1.5% du capital
                },
                'ALTCOINS': {
                    'min_volatility': 0.015,
                    'max_volatility': 0.15,
                    'min_volume_ratio': 1.3,
                    'max_spread': 0.08,
                    'position_size': 0.0125
                },
                'DEFI': {
                    'min_volatility': 0.02,
                    'max_volatility': 0.2,
                    'min_volume_ratio': 1.4,
                    'max_spread': 0.1,
                    'position_size': 0.01
                },
                'NFT': {
                    'min_volatility': 0.02,
                    'max_volatility': 0.2,
                    'min_volume_ratio': 1.4,
                    'max_spread': 0.1,
                    'position_size': 0.01
                },
                'BTC_PAIRS': {
                    'min_volatility': 0.008,
                    'max_volatility': 0.08,
                    'min_volume_ratio': 1.2,
                    'max_spread': 0.03,
                    'position_size': 0.0175
                }
            },
            'markets': {
                'MAJOR': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
                'ALTCOINS': ['DOT/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT'],
                'DEFI': ['UNI/USDT', 'AAVE/USDT', 'LINK/USDT', 'SUSHI/USDT', 'CRV/USDT'],
                'NFT': ['SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'ENJ/USDT'],
                'BTC_PAIRS': ['ETH/BTC', 'ADA/BTC', 'XRP/BTC', 'DOT/BTC']
            },
            'backtest': {
                'test_duration_hours': 24,
                'configurations': {
                    'conservative': {
                        'risk_per_trade': 0.01,
                        'max_trades_per_hour': 200,
                        'max_exposure': 0.1
                    },
                    'balanced': {
                        'risk_per_trade': 0.015,
                        'max_trades_per_hour': 250,
                        'max_exposure': 0.15
                    },
                    'aggressive': {
                        'risk_per_trade': 0.02,
                        'max_trades_per_hour': 300,
                        'max_exposure': 0.2
                    }
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            }
        }
        
        # Chargement configuration
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Charge la configuration depuis les fichiers."""
        try:
            config = self.default_config.copy()
            
            # Fichier trading
            trading_file = self.config_path / 'trading_config.json'
            if trading_file.exists():
                with open(trading_file, 'r') as f:
                    trading_config = json.load(f)
                config.update(trading_config)
            
            # Fichier backtest
            backtest_file = self.config_path / 'backtest_config.json'
            if backtest_file.exists():
                with open(backtest_file, 'r') as f:
                    backtest_config = json.load(f)
                config['backtest'].update(backtest_config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Erreur chargement config: {e}")
            return self.default_config

    def get(self, key: str, default: Optional[str] = None) -> any:
        """Récupère une valeur de configuration."""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: any):
        """Modifie une valeur de configuration."""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config[k]
            config[keys[-1]] = value
        except Exception as e:
            self.logger.error(f"Erreur modification config: {e}")

    def save(self):
        """Sauvegarde la configuration."""
        try:
            # Sauvegarde trading_config.json
            trading_config = {
                'exchange': self.config['exchange'],
                'trading': self.config['trading']
            }
            trading_file = self.config_path / 'trading_config.json'
            with open(trading_file, 'w') as f:
                json.dump(trading_config, f, indent=2)
            
            # Sauvegarde backtest_config.json  
            backtest_file = self.config_path / 'backtest_config.json'
            with open(backtest_file, 'w') as f:
                json.dump(self.config['backtest'], f, indent=2)
                
            self.logger.info("Configuration sauvegardée")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde config: {e}")

# Instance globale
config = Config()