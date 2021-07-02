from typing import *
from pathlib import Path
from itertools import dropwhile
from importlib import util as importlib_util 
from io import StringIO 
import sys
import pathlib
import json
import inspect
import re


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        

def write_str(path: Path, content: str, temp=False):
    with open(path, "w") as stream:
        stream.write(content)
        

def write_json(path: Path, content: dict, temp=False):
    with open(path, "w") as stream:
        json.dump(content, stream)


def get_class_from_string(code: str, clsname: str) -> Any:
    my_name = 'stratmodule'
    my_spec = importlib_util.spec_from_loader(my_name, loader=None)
    stratmodule = importlib_util.module_from_spec(my_spec)
    exec(code, stratmodule.__dict__)
    sys.modules['stratmodule'] = stratmodule
    return getattr(stratmodule, clsname)


def get_function_body(func):
    source_lines = inspect.getsourcelines(func)[0]
    source_lines = dropwhile(lambda x: x.startswith('@'), source_lines)
    source = ''.join(source_lines)
    pattern = re.compile(r'(async\s+)?def\s+\w+\s*\(.*?\)\s*:\s*(.*)', flags=re.S)
    lines = pattern.search(source).group(2).splitlines()
    if len(lines) == 1:
        return lines[0]
    else:
        indentation = len(lines[1]) - len(lines[1].lstrip())
        return '\n'.join([lines[0]] + [line[indentation:] for line in lines[1:]])