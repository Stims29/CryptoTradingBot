import logging
from datetime import datetime

class PositionTracker:
    def __init__(self):
        self.positions = {}
        self.pnl = 0.0
        self.logger = logging.getLogger(__name__)

    def track_position(self, symbol: str, price: float, size: float, side: str):
        position_key = f"{symbol}_{side}"
        if position_key not in self.positions:
            self.positions[position_key] = {
                'entry_price': price,
                'size': size,
                'side': side,
                'timestamp': datetime.now()
            }