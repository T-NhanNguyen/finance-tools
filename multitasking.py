import functools
import threading

# A minimal mock of the `multitasking` library to satisfy yfinance's
# hard import requirement when the real library fails to build (e.g. Vercel uv --no-build)

class DummyTask:
    def join(self):
        pass

def set_max_threads(threads: int):
    pass

def task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Execute synchronously on the main thread
        # This completely side-steps Vercel/uvicorn threading limitations
        func(*args, **kwargs)
        return DummyTask()
    return wrapper

def wait_for_tasks():
    # Everything ran synchronously, so there's nothing to wait for
    pass
