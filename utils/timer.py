import time
import logging
from functools import wraps

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TIMER')

def timeit(tag: str = ""):
    """
    计时装饰器（输出到日志）
    参数:
        tag: 可自定义的阶段标识
    使用方式:
        @timeit("重投影步骤")
        def my_function(...):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.info(f"🚀 启动: {tag or func.__name__}")

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start_time
            logger.info(f"🏁 完成: {tag or func.__name__} | 耗时: {elapsed:.4f}s")
            return result

        return wrapper

    return decorator