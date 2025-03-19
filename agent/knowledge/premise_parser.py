from abc import ABC
from core.connectives import Clause

class PremiseParser(ABC):
  def parse(self, premises: list[str]) -> list[Clause]:
        pass