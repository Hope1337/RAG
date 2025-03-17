from dataclasses import dataclass
from .types import Variable, Clause

@dataclass
class Exists:
    """Logical Existential Quantifier"""

    variable: Variable
    body: "Clause"

    def __str__(self) -> str:
        return f"∃{str(self.variable)}({str(self.body)})"


@dataclass
class All:
    """Logical Universal Quantifier"""

    variable: Variable
    body: "Clause"

    def __str__(self) -> str:
        return f"∀{str(self.variable)}({str(self.body)})"
