from abc import ABC
from core.types import Clause

class QuestionParser(ABC):
  def parse(self, sentences: list[str]) -> list[Clause]:
        pass