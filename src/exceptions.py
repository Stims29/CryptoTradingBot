#!/usr/bin/env python
from typing import Dict, Optional


class TradingError(Exception):
    """Exception de base pour le trading."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class StrategyError(TradingError):
    """Exception liée aux stratégies de trading."""
    pass


class PositionError(TradingError):
    """Exception liée à la gestion des positions."""
    pass


class OrderError(TradingError):
    """Exception liée aux ordres de trading."""
    pass


class RiskError(TradingError):
    """Exception liée à la gestion des risques."""
    pass


class DataError(TradingError):
    """Exception liée aux données de marché."""
    pass


class ConfigError(TradingError):
    """Exception liée à la configuration."""
    pass


class BotError(TradingError):
    """Exception liée au bot de trading."""
    pass


class MarketError(TradingError):
    """Exception liée aux conditions de marché."""
    def __init__(
        self,
        message: str,
        symbol: str,
        condition: str,
        details: Optional[Dict] = None
    ):
        self.symbol = symbol
        self.condition = condition
        super().__init__(message, details)


class ValidationError(TradingError):
    """Exception liée à la validation des données/paramètres."""
    def __init__(
        self,
        message: str,
        field: str,
        value: any,
        details: Optional[Dict] = None
    ):
        self.field = field
        self.value = value
        super().__init__(message, details)


class NetworkError(TradingError):
    """Exception liée aux problèmes réseau."""
    def __init__(
        self,
        message: str,
        endpoint: str,
        status_code: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        self.endpoint = endpoint
        self.status_code = status_code
        super().__init__(message, details)


class ExchangeError(TradingError):
    """Exception liée à l'exchange."""
    def __init__(
        self,
        message: str,
        exchange: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        self.exchange = exchange
        self.error_code = error_code
        super().__init__(message, details)


class AuthenticationError(TradingError):
    """Exception liée à l'authentification."""
    def __init__(
        self,
        message: str,
        credentials: str,
        details: Optional[Dict] = None
    ):
        self.credentials = credentials
        super().__init__(message, details)


class RateLimitError(TradingError):
    """Exception liée aux limites de taux."""
    def __init__(
        self,
        message: str,
        limit: int,
        reset_time: int,
        details: Optional[Dict] = None
    ):
        self.limit = limit
        self.reset_time = reset_time
        super().__init__(message, details)


class InsufficientFundsError(TradingError):
    """Exception liée au manque de fonds."""
    def __init__(
        self,
        message: str,
        required: float,
        available: float,
        details: Optional[Dict] = None
    ):
        self.required = required
        self.available = available
        super().__init__(message, details)


class TimeoutError(TradingError):
    """Exception liée aux délais d'attente."""
    def __init__(
        self,
        message: str,
        operation: str,
        timeout: float,
        details: Optional[Dict] = None
    ):
        self.operation = operation
        self.timeout = timeout
        super().__init__(message, details)