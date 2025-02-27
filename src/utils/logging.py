#!/usr/bin/env python
# src/logger.py

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure un logger standard."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

class DetailedLogger:
    def __init__(self, name: str = None):
        """Initialise le logger avec configuration avancée."""
        self.name = name or 'detailed'
        self.logger = self._setup_logger()
    
    def critical(self, msg: str):
        """Enregistre un message de niveau critique dans les logs."""
        if hasattr(self, 'logger'):
            self.logger.critical(msg)
        else:
            # Fallback si logger n'est pas disponible
            print(f"CRITICAL: {msg}")
        
    def _setup_logger(self) -> logging.Logger:
        """Configure le logger avec handlers console et fichier."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Création dossier logs si nécessaire
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Handler fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            f"logs/{self.name}_{timestamp}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Ajout handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

    # API Standard Logging
    def info(self, msg: str):
        """Log niveau INFO."""
        self.logger.info(msg)

    def error(self, msg: str):
        """Log niveau ERROR."""
        self.logger.error(msg)

    def warning(self, msg: str):
        """Log niveau WARNING."""
        self.logger.warning(msg)

    def debug(self, msg: str):
        """Log niveau DEBUG."""
        self.logger.debug(msg)

    # API Étendue
    def log_signal(self, data: Dict):
        """Log un signal de trading."""
        self.logger.debug(
            f"SIGNAL:\n" + 
            "\n".join(f"  {k}: {v}" for k, v in data.items())
        )

    def log_trade(self, action: str, data: Dict):
        """Log une action de trading."""
        self.logger.info(
            f"TRADE {action}:\n" + 
            "\n".join(f"  {k}: {v}" for k, v in data.items())
        )

    def log_error(self, error: str, context: Optional[Dict] = None):
        """Log une erreur avec contexte optionnel."""
        error_msg = f"ERROR: {error}"
        if context:
            error_msg += "\nContext:\n" + "\n".join(f"  {k}: {v}" for k, v in context.items())
        self.logger.error(error_msg)

    def log_position(self, action: str, data: Dict):
        """Log une action sur une position."""
        self.logger.info(
            f"POSITION {action}:\n" + 
            "\n".join(f"  {k}: {v}" for k, v in data.items())
        )

    def log_metrics(self, metrics: Dict):
        """Log les métriques de performance."""
        self.logger.info(
            f"METRICS:\n" + 
            "\n".join(f"  {k}: {v}" for k, v in metrics.items())
        )

    def log_market(self, symbol: str, data: Dict):
        """Log des données de marché."""
        self.logger.debug(
            f"MARKET {symbol}:\n" + 
            "\n".join(f"  {k}: {v}" for k, v in data.items())
        )

    def close(self):
        """Ferme proprement les handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)