from importlib import util as importlib_util
from io import StringIO
from itertools import dropwhile
from pathlib import Path
from typing import Any
from functools import wraps

import inspect
import json
import re
import sys
import arrow
import logging
import time
import gc

logger = logging.getLogger(__name__)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def write_str(path: Path, content: str, temp=False):
    with open(path, "w") as stream:
        stream.write(content)


def write_json(path: Path, content: dict, temp=False):
    with open(path, "w") as stream:
        json.dump(content, stream)


def get_readable_date() -> str:
    return arrow.utcnow().shift(hours=7).strftime("%Y-%m-%d_%H-%M-%S")


def get_class_from_string(code: str, clsname: str) -> Any:
    my_name = "stratmodule"
    my_spec = importlib_util.spec_from_loader(my_name, loader=None)

    if my_spec is None:
        raise ImportError(f"Module {my_name} not found")

    stratmodule = importlib_util.module_from_spec(my_spec)
    exec(code, stratmodule.__dict__)
    sys.modules["stratmodule"] = stratmodule
    return getattr(stratmodule, clsname)


def get_function_body(func):
    source_lines = inspect.getsourcelines(func)[0]
    source_lines = dropwhile(lambda x: x.startswith("@"), source_lines)
    source = "".join(source_lines)
    pattern = re.compile(r"(async\s+)?def\s+\w+\s*\(.*?\)\s*:\s*(.*)", flags=re.S)
    lines = pattern.search(source).group(2).splitlines()
    if len(lines) == 1:
        return lines[0]
    else:
        indentation = len(lines[1]) - len(lines[1].lstrip())
        return "\n".join([lines[0]] + [line[indentation:] for line in lines[1:]])


def log_execute_time(name: str = None):
    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(*args, **kwargs):
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                end = time.time() - start
                logger.info('"{}" executed in {:.2f}s'.format(
                    name if name is not None else function.__name__, end
                ))
        return somedec_inner
    return somedec_outer


def free_mem(var):
    del var
    gc.collect()