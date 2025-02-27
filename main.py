# main.py
import logging
import os
from src.core.bot import TradingBot
from src.utils.logger import Logger
from config.settings import Settings

def setup_environment():
    """Configure l'environnement d'exécution"""
    # Création des dossiers nécessaires
    os.makedirs('logs', exist_ok=True)
    
    # Configuration du logger
    Logger.setup_logger(
        'trading_bot',
        'logs/trading.log',
        level=logging.INFO
    )

def main():
    """Point d'entrée principal de l'application"""
    try:
        # Configuration de l'environnement
        setup_environment()
        
        # Création et démarrage du bot
        bot = TradingBot()
        bot.run()
        
    except KeyboardInterrupt:
        logging.info("Arrêt manuel du bot...")
        if 'bot' in locals():
            bot.shutdown()
    except Exception as e:
        logging.error(f"Erreur critique: {str(e)}")
        if 'bot' in locals():
            bot.shutdown()

if __name__ == "__main__":
    main()