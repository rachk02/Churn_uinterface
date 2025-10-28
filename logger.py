"""
Module de logging pour l'application
"""
import logging
import os
from datetime import datetime


def setup_logger(name='churn_app'):
    """Configure le système de logging"""

    # Créer le dossier logs s'il n'existe pas
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Éviter les doublons de handlers
    if logger.handlers:
        return logger

    # Handler pour fichier
    log_filename = f'logs/app_{datetime.now():%Y%m%d}.log'
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Handler pour console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# Logger global
logger = setup_logger()
