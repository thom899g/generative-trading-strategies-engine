"""
Real-time market data pipeline with caching, validation, and preprocessing.
Integrates with CCXT for exchange data and Firebase for state management.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from firebase_admin import firestore
import firebase_admin
from firebase_admin import credentials

from config import config

logger = logging.getLogger(__name__)

class DataPipeline:
    """Real-time market data pipeline with caching and preprocessing"""
    
    def __init__(self):
        self.exchange = self._init_exchange()
        self.db = self._init_firebase()
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange with error handling"""
        try:
            exchange_class = getattr(ccxt, config.exchange_id)
            exchange = exchange_class({
                'apiKey': config.api_key,
                'secret': config.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Test connection
            exchange.load_markets()
            logger.info(f"Connected to {config.exchange_id} exchange")
            return exchange
            
        except AttributeError as e:
            logger.error(f"Exchange {config.exchange_id} not found in CC