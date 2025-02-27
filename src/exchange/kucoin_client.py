# src/exchange/kucoin_client.py
import ccxt
import time
import logging
from typing import Dict, Any, List, Optional

class KuCoinClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.markets = {}
        self.initialize_exchange()

    def initialize_exchange(self):
        """Initialise la connexion à KuCoin"""
        try:
            self.exchange = ccxt.kucoin({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            self.markets = self.exchange.load_markets()
            self.logger.info("KuCoin client initialisé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de KuCoin: {str(e)}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100, since: Optional[int] = None) -> List[List]:
        """Récupère les données OHLCV"""
        try:
            if self.exchange is None:
                self.initialize_exchange()

            # Formater le symbole pour KuCoin
            formatted_symbol = symbol.replace('/', '')
            
            # Récupérer les données avec retry
            for attempt in range(3):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=formatted_symbol,
                        timeframe=timeframe,
                        limit=limit,
                        since=since
                    )
                    return ohlcv
                except Exception as e:
                    if attempt == 2:  # Dernière tentative
                        raise
                    time.sleep(1)  # Attendre avant de réessayer
                    
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données OHLCV pour {symbol}: {str(e)}")
            return []

    def close(self):
        """Ferme la connexion"""
        self.exchange = None
        self.markets = {}