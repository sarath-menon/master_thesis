import time
import hashlib
import functools
from typing import Dict, Tuple, Any, Callable

def cache_result(expiration_time: int = 300):
    def decorator(func: Callable):
        cache: Dict[str, Tuple[Any, float]] = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = _generate_cache_key(*args, **kwargs)

            cached_result = cache.get(cache_key)
            if cached_result:
                result, timestamp = cached_result
                if time.time() - timestamp < expiration_time:
                    print("Using cached result")
                    return result

            result = await func(*args, **kwargs)

            cache[cache_key] = (result, time.time())
            # print("Caching new result")

            return result

        def clear_cache():
            cache.clear()

        wrapper.clear_cache = clear_cache
        return wrapper

    return decorator

def _generate_cache_key(*args, **kwargs) -> str:
    key = hashlib.md5()
    for arg in args:
        key.update(str(arg).encode())
    for k, v in kwargs.items():
        key.update(str(k).encode())
        key.update(str(v).encode())
    return key.hexdigest()