from core.atom import Atom
from core.variable import Variable
from core.constant import Constant
from core.connectives import All, Exists, And, Or, Not, Implies, Clause
from core.predicate import Predicate
from knowledge.premise_parser import PremiseParser
from nltk.sem.logic import (
    LogicParser, Expression,
    AllExpression, ExistsExpression,
    ImpExpression, AndExpression, OrExpression, NegatedExpression, 
    ApplicationExpression, ConstantExpression
  )

symbol_replacements = {
    'exists ': ['∃'],
    'forall ': ['∀'],
    '& ': ['∧'],
    '| ': ['∨'],
    '! ': ['¬'],
    ' => ': ['→']
}

class PremiseFolParser(PremiseParser):
  """
  Parse a premise which under First Order Logic. 
  Example: 
    "∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))"
    "∀x (DrinkRegularly(x, coffee) ∨ (¬WantToBeAddictedTo(x, caffeine)))"
    "∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))"
    "¬(Student(rina) ⊕ ¬AwareThatDrug(rina, caffeine))"
    "¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))"
  """

  nltk_logic_parser: LogicParser

  def __init__(self):
      super().__init__()
      self.nltk_logic_parser = LogicParser()


  def parse(self, folLogics: list[str]) -> list[Clause]:
        clauses = []
        for folLogic in folLogics:
          try:
            nor_fol = normalize_folText(folLogic)
            exp = self.nltk_logic_parser.parse(nor_fol)
            storageVariables =[]
            clauses.append(convert_to_exp_logic(exp, storageVariables))
          except Exception as e:
            SyntaxError(f"Cannot parse First Order Logic: {folLogic}")  
        return clauses
  
def normalize_folText(folText):
  # process for XOR operator 
  folText = process_xor_operator(folText)
  
  # replace symbols
  for nltk_token, symbols in symbol_replacements.items():
      for symbol in symbols:
          folText = folText.replace(symbol, nltk_token)
  return folText

def process_xor_operator(folText: str):
  xor_char = '⊕'
  index = folText.find(xor_char)
  if index == -1:
      return folText
  left_start = get_left_xor_start(folText, index)
  right_end = get_right_xor_end(folText, index)

  left = folText[left_start:index].strip()
  right = folText[index+1: right_end].strip()
  
  not_char = '¬'
  if left[0] == not_char:
      not_left = left[1:]
  else:
      not_left = '¬' + left
  if right[0] == not_char:
      not_right = right[1:]
  else:
      not_right = '¬' + right
      
  xor_formula = f'( ({left} ∧ {not_right}) ∨ ({not_left} ∧ {right}) )'
  folText = folText[0:left_start] + xor_formula + folText[right_end:]
  folText = process_xor_operator(folText)
  return folText

def get_left_xor_start(text, index):
  close_parentheses_count = 0
  for i in range(index, 0, -1):
      if text[i] == ')':
          close_parentheses_count +=1
      
      if text[i] == '(':
          if close_parentheses_count == 0:
              return i+1
          else:
              close_parentheses_count -=1
      
      if text[i] in ['∧', '∨', '→']:
          return i+1
  return 0

def get_right_xor_end(text, index):
  open_parentheses_count = 0
  for i in range(index, len(text), 1):
      if text[i] == ' ':
          continue
      if text[i] == '(':
          open_parentheses_count +=1
      
      if text[i] == ')':
          if open_parentheses_count == 0:
              return i
          else:
              open_parentheses_count -=1
      
      if text[i] in ['∧', '∨', '→']:
          return i-1
  return len(text)

def convert_to_exp_logic(exp: Expression, storageVariables: list[Variable]) -> Clause: 
  if isinstance(exp, AllExpression):
      variable = Variable(exp.variable.name)
      storageVariables.append(variable)
      return All(
          variable = variable,
          body= convert_to_exp_logic(exp.term, storageVariables)
      )
  elif isinstance(exp, ExistsExpression):
      variable = Variable(exp.variable.name)
      storageVariables.append(variable)
      return Exists(
          variable = variable,
          body= convert_to_exp_logic(exp.term, storageVariables)
      )
  elif isinstance(exp, ImpExpression):
      return Implies(
          antecedent= convert_to_exp_logic(exp.first, storageVariables),
          consequent= convert_to_exp_logic(exp.second, storageVariables),
      )
  elif isinstance(exp, AndExpression):
      return And(
          convert_to_exp_logic(exp.first, storageVariables),
          convert_to_exp_logic(exp.second, storageVariables),
      )
  elif isinstance(exp, OrExpression):
      return Or(
          convert_to_exp_logic(exp.first, storageVariables),
          convert_to_exp_logic(exp.second, storageVariables),
      )
  elif isinstance(exp, NegatedExpression):
      return Not(
          convert_to_exp_logic(exp.term, storageVariables),
      )
  elif isinstance(exp, ApplicationExpression):
      predicate_exp, args = exp.uncurry()
      assert isinstance(predicate_exp, ConstantExpression)
      predicate = predicate_exp.variable
      terms = []
      for arg in args:
          if isinstance(arg, ConstantExpression):
              terms.append(Constant(
                  symbol= arg.variable.name
              ))
          else:
              variable_name = arg.variable.name
              existing_variables = list(filter(lambda x: x.name == variable_name, storageVariables))
              if len(existing_variables) > 0 :
                  terms.append(existing_variables[0])
              else:
                  terms.append(Variable(variable_name))

      return Atom(
          predicate= Predicate(
              symbol=predicate.name
          ),
          terms= terms
      )

