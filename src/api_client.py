#!/usr/bin/env python
import logging
import hmac
import hashlib
import time
import aiohttp
import asyncio
from typing import Dict, Optional
from urllib.parse import urlencode

from .exceptions import APIError
from .config import config
from .logger import setup_logging

class APIClient:
    def __init__(self):
        """Initialise le client API."""
        self.logger = setup_logging(__name__)
        self.session = None
        self.base_url = 'https://api.kucoin.com'
        
        # Credentials
        self.api_key = config.get('exchange.api_key')
        self.api_secret = config.get('exchange.api_secret')
        self.passphrase = config.get('exchange.password')
        
        # Rate limiting
        self.rate_limits = {
            'public': {'rate': 10, 'burst': 20},
            'private': {'rate': 5, 'burst': 10}
        }
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms

    async def __aenter__(self):
        """Context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Context manager exit."""
        await self.close()

    async def close(self):
        """Ferme la session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, data: str = '') -> str:
        """Génère la signature pour les requêtes authentifiées."""
        message = timestamp + method + endpoint + data
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_headers(self, method: str, endpoint: str, data: str = '') -> Dict:
        """Prépare les headers de la requête."""
        timestamp = str(int(time.time() * 1000))
        headers = {
            'KC-API-KEY': self.api_key,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-PASSPHRASE': self.passphrase,
            'KC-API-SIGN': self._generate_signature(timestamp, method, endpoint, data),
            'Content-Type': 'application/json'
        }
        return headers

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Gère la réponse de l'API."""
        try:
            data = await response.json()
            
            if response.status != 200:
                error_msg = data.get('msg', 'Unknown API error')
                raise APIError(
                    f"API error ({response.status}): {error_msg}",
                    {'status': response.status, 'data': data}
                )
                
            return data
            
        except aiohttp.ContentTypeError:
            text = await response.text()
            raise APIError(
                f"Invalid JSON response: {text}",
                {'status': response.status}
            )

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict:
        """Exécute une requête API."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            # Préparation URL
            url = self.base_url + endpoint
            if params:
                url += '?' + urlencode(params)
                
            # Headers
            headers = {}
            if auth:
                headers = self._get_headers(method, endpoint, str(data or ''))
            
            # Requête
            async with self.session.request(
                method,
                url,
                headers=headers,
                json=data
            ) as response:
                result = await self._handle_response(response)
                self.last_request_time = time.time()
                return result
                
        except aiohttp.ClientError as e:
            raise APIError(f"Request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise APIError("Request timed out")

    async def get_ticker(self, symbol: str) -> Dict:
        """Récupère le ticker d'un symbole."""
        return await self._request(
            'GET',
            f'/api/v1/market/orderbook/level1?symbol={symbol}'
        )

    async def get_markets(self) -> Dict:
        """Récupère la liste des marchés."""
        return await self._request('GET', '/api/v1/symbols')

    async def get_account(self) -> Dict:
        """Récupère les informations du compte."""
        return await self._request(
            'GET',
            '/api/v1/accounts',
            auth=True
        )

    async def create_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None
    ) -> Dict:
        """Crée un ordre."""
        data = {
            'symbol': symbol,
            'side': side,
            'size': str(size)
        }
        
        if price:
            data['price'] = str(price)
            data['type'] = 'limit'
        else:
            data['type'] = 'market'
            
        return await self._request(
            'POST',
            '/api/v1/orders',
            data=data,
            auth=True
        )

    async def cancel_order(self, order_id: str) -> Dict:
        """Annule un ordre."""
        return await self._request(
            'DELETE',
            f'/api/v1/orders/{order_id}',
            auth=True
        )

    async def get_order(self, order_id: str) -> Dict:
        """Récupère les détails d'un ordre."""
        return await self._request(
            'GET',
            f'/api/v1/orders/{order_id}',
            auth=True
        )