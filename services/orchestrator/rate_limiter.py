"""
Rate limiting utilities for the AgenticRAG system
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_second: float
    burst_limit: int = 1

class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self):
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': 0,
            'last_update': time.time(),
            'limit': RateLimit(1.0)  # Default: 1 request per second
        })
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    def set_rate_limit(self, key: str, limit: RateLimit):
        """Set rate limit for a specific key"""
        self.buckets[key]['limit'] = limit
        self.buckets[key]['tokens'] = limit.burst_limit
    
    async def acquire(self, key: str) -> bool:
        """Acquire a token for the given key, waiting if necessary"""
        async with self.locks[key]:
            bucket = self.buckets[key]
            limit = bucket['limit']
            
            # Add tokens based on time passed
            now = time.time()
            elapsed = now - bucket['last_update']
            new_tokens = elapsed * limit.requests_per_second
            bucket['tokens'] = min(bucket['tokens'] + new_tokens, limit.burst_limit)
            bucket['last_update'] = now
            
            # If we have tokens, consume one
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            else:
                # Calculate wait time needed
                wait_time = (1 - bucket['tokens']) / limit.requests_per_second
                await asyncio.sleep(wait_time)
                bucket['tokens'] -= 1
                bucket['last_update'] = time.time()
                return True

# Global rate limiter instance
rate_limiter = RateLimiter()