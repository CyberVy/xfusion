import functools
import time
from typing import List,Union,Optional

class UIMixin:
    overrides = ["load_ui"]

    def load_ui(self,**kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'load_ui'")


def lists_append(element,lists):
    for _list in lists:
        _list.append(element)

def lock(lock_state:Optional[List[Optional[bool]]] = None):
    lock_state = lock_state if lock_state is not None else [False,None]

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args,**kwargs):
            if lock_state[0]:
                raise RuntimeError(f"Async and multiple threads are not allowed for {f.__name__}, because {lock_state[1]} is working.")
            try:
                lock_state[0] = True
                lock_state[1] = f.__name__
                return f(*args,**kwargs)
            finally:
                lock_state[0] = False
                lock_state[1] = None

        return wrapper

    return decorator

def safe_block():
    import time
    while (1):
        try:
            time.sleep(len("hello world. " * 100))
        except KeyboardInterrupt:
            break
    
