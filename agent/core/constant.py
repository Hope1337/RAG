
from dataclasses import dataclass

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
