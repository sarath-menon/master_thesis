import time
import hashlib
from typing import Dict, Tuple, Callable, Any
from functools import wraps

# Cache to store recent predictions
prediction_cache: Dict[str, Tuple[Any, float]] = {}
CACHE_EXPIRATION_TIME = 3000  # 50 minutes in seconds

def cache_prediction(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        enable_cache = kwargs.get('enable_cache', True)
        reset_cache = kwargs.get('reset_cache', False)

        if reset_cache:
            prediction_cache.clear()
            return {"message": "Cache has been reset"}

        if not enable_cache:
            return await func(*args, **kwargs)

        cache_key = generate_cache_key(*args, **kwargs)

        cached_result = prediction_cache.get(cache_key)
        if cached_result:
            prediction, timestamp = cached_result
            if time.time() - timestamp < CACHE_EXPIRATION_TIME:
                return prediction

        response = await func(*args, **kwargs)

        prediction_cache[cache_key] = (response, time.time())

        return response

    return wrapper

def generate_cache_key(*args, **kwargs) -> str:
    key = hashlib.md5()
    for arg in args:
        if hasattr(arg, 'filename'):
            key.update(arg.filename.encode())
        elif arg is not None:
            key.update(str(arg).encode())
    for k, v in kwargs.items():
        if v is not None:
            key.update(f"{k}:{v}".encode())
    return key.hexdigest()