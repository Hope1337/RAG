from core.types import Clause
from .premise_parser import PremiseParser

class PremiseSentenceParser(PremiseParser):
  """
  Parse a premise which under Nature language 
  """
  def parse(self, sentences: list[str]) -> list[Clause]:
        pass