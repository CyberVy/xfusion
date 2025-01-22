import functools


class UIMixin:
    overrides = ["load_ui"]

    def load_ui(self,**kwargs):
        raise NotImplementedError(f"{object.__getattribute__(self, '__class__')} not implement 'load_ui'")


def lists_append(element,lists):
    for _list in lists:
        _list.append(element)

def lock(lock_state=None):
    lock_state = lock_state if lock_state is not None else [False]

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args):
            if lock_state[0]:
                raise RuntimeError(f"Async and multiple threads are not allowed for {f.__name__}.")
            try:
                lock_state[0] = True
                return f(*args)
            finally:
                lock_state[0] = False

        return wrapper

    return decorator
