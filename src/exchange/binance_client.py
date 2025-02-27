from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from datetime import datetime
import logging
from typing import Dict

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """Initialise le client Binance."""
        self.logger = logging.getLogger(__name__)
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        
    def get_historical_data(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Récupère les données historiques."""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Conversion types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Définir timestamp comme index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except BinanceAPIException as e:
            self.logger.error(f"Erreur API Binance: {e}")
            return pd.DataFrame()
            
    def get_account_balance(self) -> float:
        """Récupère le solde du compte."""
        try:
            account = self.client.get_account()
            return float(
                next(
                    asset['free'] 
                    for asset in account['balances'] 
                    if asset['asset'] == 'USDT'
                )
            )
        except BinanceAPIException as e:
            self.logger.error(f"Erreur balance: {e}")
            return 0.0
            
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = None,
        order_type: str = 'MARKET'
    ) -> Dict:
        """Place un ordre."""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type == 'LIMIT':
                params['price'] = price
                params['timeInForce'] = 'GTC'
                
            order = self.client.create_order(**params)
            
            return {
                'id': order['orderId'],
                'status': order['status'],
                'filled': float(order['executedQty']),
                'price': float(order['price']) if order['price'] else None,
                'cost': float(order.get('cummulativeQuoteQty', 0))
            }
            
        except BinanceAPIException as e:
            self.logger.error(f"Erreur ordre: {e}")
            return {}