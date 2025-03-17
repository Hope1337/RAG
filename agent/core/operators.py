from dataclasses import dataclass
from .types import Clause

@dataclass
class And:
    """Logical Conjunction"""

    args: tuple["Clause", ...]

    def __init__(self, *args: "Clause") -> None:
        # automatically reduce repeated ANDs
        simplified_args: list["Clause"] = []
        for arg in args:
            if type(arg) is And:
                simplified_args.extend(arg.args)
            else:
                simplified_args.append(arg)
        self.args = tuple(simplified_args)

    def __str__(self) -> str:
        arg_strs = []
        for arg in self.args:
            if type(arg) in [Or, Implies]:
                arg_strs.append(f"({str(arg)})")
            else:
                arg_strs.append(str(arg))
        return f"{' ∧ '.join(arg_strs)}"


@dataclass
class Or:
    """Logical Disjunction"""

    args: tuple["Clause", ...]

    def __init__(self, *args: "Clause") -> None:
        # automatically reduce repeated ORs
        simplified_args: list["Clause"] = []
        for arg in args:
            if type(arg) is Or:
                simplified_args.extend(arg.args)
            else:
                simplified_args.append(arg)
        self.args = tuple(simplified_args)

    def __str__(self) -> str:
        arg_strs = []
        for arg in self.args:
            if type(arg) in [And, Implies]:
                arg_strs.append(f"({str(arg)})")
            else:
                arg_strs.append(str(arg))
        return f"{' ∨ '.join(arg_strs)}"


@dataclass
class Not:
    """Logical Negation"""

    body: "Clause"

    def __str__(self) -> str:
        if type(self.body) is And:
            return f"¬({str(self.body)})"
        return f"¬{str(self.body)}"


@dataclass
class Implies:
    """Logical Implication"""

    antecedent: "Clause"
    consequent: "Clause"

    def __str__(self) -> str:
        antecedent_str = str(self.antecedent)
        consequent_str = str(self.consequent)
        if type(self.antecedent) in [And, Or, Implies]:
            antecedent_str = f"({antecedent_str})"
        if type(self.consequent) in [And, Or, Implies]:
            consequent_str = f"({consequent_str})"
        return f"{antecedent_str} → {consequent_str}"
