import json
import hashlib
import functools
from typing import Callable, Optional
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


def cache_key(*args, **kwargs) -> str:
    raw = json.dumps({"args": str(args), "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def cached(ttl: int = 300, prefix: str = "moviroo"):
    """
    Async caching decorator for FastAPI route handlers.
    Requires `request: Request` as first param in the decorated function.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            redis = request.app.state.redis
            key = f"{prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"
            try:
                cached_val = await redis.get(key)
                if cached_val:
                    logger.debug(f"Cache HIT: {key}")
                    return json.loads(cached_val)
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")

            result = await func(request, *args, **kwargs)

            try:
                await redis.setex(key, ttl, json.dumps(result, default=str))
            except Exception as e:
                logger.warning(f"Redis SET failed: {e}")

            return result
        return wrapper
    return decorator


async def invalidate_cache(redis, pattern: str):
    """Delete all keys matching pattern."""
    keys = await redis.keys(pattern)
    if keys:
        await redis.delete(*keys)
