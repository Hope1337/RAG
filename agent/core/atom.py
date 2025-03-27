from __future__ import annotations
from dataclasses import dataclass
from .predicate import Predicate
from .term import Term


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