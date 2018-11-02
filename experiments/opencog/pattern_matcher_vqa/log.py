import os
import functools
import logging
import inspect
import time
import datetime
import threading
import traceback
import decorator


tls = threading.local()


class CallLogger():
    def __init__(self, logger):
        self.storage = tls
        self.separator = '|'
        self.logger = logger

    def extract_argspec(self, func, args, kwargs):
        argnames = list(inspect.signature(func).parameters.keys())
        arg_slice = 1 if argnames[0] == 'self' else 0
        s_args = ', '.join(self.loggable(x) for x in args[arg_slice:])
        s_kwargs = ', '.join(self.loggable(k) + '=' + self.loggable(v) for k, v in kwargs.items())
        result = s_args + ', ' * int(bool(s_kwargs)) +  s_kwargs
        return result

    def __call__(self, fn):
        # todo: asyncio wrapper
        def sync_wrapper(func, *args, **kwargs):
            if not hasattr(self.storage, 'indent'):
                self.storage.indent = 0
            self.logger.info(self.prefix + '{}({})'.format(func.__name__, 
                                                           self.extract_argspec(func, args, kwargs)))
            self.storage.indent += 1
            trace = traceback.extract_stack(limit=4)
            self.logger.info(self.prefix +
                             'From {}:{}'.format(os.path.basename(trace[-3][0]), trace[-3][1]))

            start = time.time()

            try:
                result = func(*args, **kwargs)
                end = time.time()
                self.logger.debug(self.prefix +
                                  'Duration: {}, Result: {}'.format(datetime.timedelta(seconds=end - start), 
                                                                    self.loggable(result)))
            except Exception:
                end = time.time()
                self.logger.exception(self.prefix +
                        'Duration: {}, Result: <Exception>, '.format(datetime.timedelta(seconds=end - start)))
                raise
            finally:
                self.storage.indent -= 1

            return result

        return decorator.decorate(fn, sync_wrapper)

    @property
    def prefix(self):
        return 'Call: {}'.format(self.storage.indent * self.separator)

    @staticmethod
    def loggable(arg):
        s_arg = repr(arg)
        if len(s_arg) > 100:
            s_arg = s_arg[:100]
        return '{0}'.format(s_arg)


def logged(*args, **kwargs):
    root_logger = logging.getLogger()
    logger = kwargs.get('logger', root_logger)
    # option 1, logged used as @logged(logger=some_logger)
    if not args:
        return functools.partial(logged, logger=logger)
    # default case
    # todo: pass logger level
    return CallLogger(logger)(*args)

# example usage
if __name__ == '__main__':

    @logged
    def foo(a, **kwargs):
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-6s %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    @logged(logger=logger)
    def foor(a, **kwargs):
        foo(a, **kwargs)

    foor('test', key='value')

