from abc import ABC
from core.types import Clause
from .premise_parser import PremiseParser

class PremiseFolParser(PremiseParser):
  """
  Parse a premise which under First Order Logic 
  """
  def parse(self, folLogics: list[str]) -> list[Clause]:
        pass