from dataclasses import dataclass
from typing import Union, Optional, Any
from .operators import Implies, And, Or, Not
from .quantifiers import All, Exists


@dataclass(frozen=True)
class Constant:
    """
    A constant symbol in a logical formula.
    """

    symbol: str

    def __hash__(self) -> int:
        return hash(self.symbol)

    def __str__(self) -> str:
        return self.symbol

@dataclass(frozen=True)
class Variable:
    """
    A variable symbol in a logical formula.
    """

    name: str

    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class Predicate:
    """
    A predicate symbol in a logical formula.
    """

    symbol: str

    def __hash__(self) -> int:
        return hash(self.symbol)

    # shorthand for creating an Atom out of this predicate and terms
    def __call__(self, *terms: Constant | Variable) -> Atom:
        return Atom(self, terms)

    def __str__(self) -> str:
        return self.symbol

@dataclass(frozen=True)
class Atom:
    """
    Logical Atom (predicate with terms)
    """

    predicate: Predicate
    terms: tuple[Term, ...]

    def __str__(self) -> str:
        terms_str = ",".join([str(term) for term in self.terms])
        return f"{self.predicate.symbol}({terms_str})"

Term = Union[Constant, Variable]
Clause = Union[Atom, Implies, Not, And, Or, Exists, All]