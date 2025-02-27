# src/exchange/base_client.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class BaseExchangeClient(ABC):
    """Classe abstraite définissant l'interface pour les clients d'exchange"""
    
    @abstractmethod
    def connect(self):
        """Établit la connexion avec l'exchange"""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Ferme la connexion avec l'exchange"""
        pass
        
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[List]:
        """Récupère les données OHLCV"""
        pass
        
    @abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Crée un ordre sur l'exchange"""
        pass
        
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Récupère le solde du compte"""
        pass
        
    @abstractmethod
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Récupère les ordres ouverts"""
        pass