import functools
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from types import FunctionType, MethodType
from typing import Union, Optional, Callable, Any

executor = ThreadPoolExecutor()


def naive_async(f: Callable[..., Any]) -> Callable[..., Future]:
    """Execute function in thread pool executor and return Future."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return executor.submit(f, *args, **kwargs)
    return wrapper


def retry(n: int = 3) -> Callable:
    """Retry function execution on error up to n times."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Retry count must be an integer >= 1.")

    def decorator(f: Callable[..., Any]):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(1, n + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    print(f"Warning: {f.__name__} attempt {i}/{n} failed: {e}")
            raise last_err
        return wrapper
    return decorator


def threads_execute(
    f: Union[FunctionType, MethodType],
    args: Union[list, tuple],
    _await: bool = True
) -> list:
    """Execute function across threads for given arguments."""
    threads = []
    if _await:
        for arg in args[1:]:
            thread = threading.Thread(target=f, args=(arg,))
            threads.append(thread)
            thread.start()
        f(args[0])
        for thread in threads:
            thread.join()
    else:
        for arg in args:
            thread = threading.Thread(target=f, args=(arg,))
            threads.append(thread)
            thread.start()
    return threads


def lock(lock_state: Optional[list] = None):
    """Simple thread-concurrency guard decorator."""
    state = lock_state if lock_state is not None else [False, None]

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if state[0]:
                raise RuntimeError(
                    f"Concurrency error: {f.__name__} blocked because {state[1]} is active."
                )
            try:
                state[0] = True
                state[1] = f.__name__
                return f(*args, **kwargs)
            finally:
                state[0] = False
                state[1] = None
        return wrapper
    return decorator
