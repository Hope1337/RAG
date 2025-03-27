from typing import Any
from core.connectives import Clause
from .premise_parser import PremiseParser
from transition_amr_parser.parse import AMRParser

class PremiseSentenceParser(PremiseParser):
  """
  Parse a premise which under Nature language 
  """
  amr_parser: AMRParser

  def __init__(self,
      arm_parse_model: str = 'doc-sen-conll-amr-seed42',
    ):
    super().__init__()
    self.amr_parser = AMRParser.from_pretrained(arm_parse_model)


  def parse(self, sentences: list[str]) -> list[Clause]:
    penman_notations = self.to_penman_notation(sentences);
    pass

  def to_penman_notation(self, sentences: list[str]) -> list[Any]:
    tok_sentences = []
    for sen in sentences:
      tokens, positions = self.amr_parser.tokenize(sen)
      tok_sentences.append(tokens)
    
    annotations, machines = self.amr_parser.parse_sentences(tok_sentences)
    notations = [machine.get_amr().to_penman(jamr=False, isi=True) for machine in machines]
    return notations