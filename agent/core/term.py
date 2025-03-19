from __future__ import annotations
from typing import Union

from .constant import Constant
from .variable import Variable

Term = Union[Constant, Variable]
