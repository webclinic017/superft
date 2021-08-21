import ast
from typing import List


string = """
class MyClass:
    def __init__(self):
        pass
"""


def get_classes_in_python_string(string: str) -> List[str]:
    tree = ast.parse(string)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


print(get_classes_in_python_string(string))