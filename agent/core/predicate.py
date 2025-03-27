
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constant import Constant
from .variable import Variable

if TYPE_CHECKING:
    from .atom import Atom

@dataclass(frozen=True)
class Predicate:
    """
    A predicate symbol in a logical formula.
    """

    symbol: str

    def __hash__(self) -> int:
        return hash(self.symbol)

    # shorthand for creating an Atom out of this predicate and terms
    def __call__(self, *terms: Constant | Variable) -> "Atom":
        from .atom import Atom
        return Atom(self, terms)

    def __str__(self) -> str:
        return self.symbol
