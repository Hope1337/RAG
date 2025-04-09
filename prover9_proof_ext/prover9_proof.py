from __future__ import annotations
from dataclasses import dataclass
from nltk.sem.logic import Expression, Variable, ConstantExpression, QuantifiedExpression, ApplicationExpression, BooleanExpression, NegatedExpression, IndividualVariableExpression, AbstractVariableExpression
from prover9_proof_ext.utils import detect_application_expressions, detect_quantified_variables, resolve_quantified_variables, find_same_application, detect_section
import re
import xml.etree.ElementTree as ET
from typing import Optional

class Prover9ProofStep:
  """
  Class for an proof step. Ex: Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan)
  """
  original_expr: Expression # Ex: all x y. ( Takes(x, y) & Passes(x, y) -> AcquiresKnowledge(x) )
  quantified_variableMap: dict[Variable, ConstantExpression] # Ex: {x -> Tuan, y -> NLP}

  def __init__(self, implication_clause: ProofClause):
    self.original_expr = implication_clause.literals[0]
    quantified_variables = detect_quantified_variables(self.original_expr)
    self.quantified_variableMap = { variable: None for variable in quantified_variables} # initial None for all quantified variables
  
  def set_variable(self, variable: Variable, constant: ConstantExpression):
    self.quantified_variableMap[variable] = constant
  
  def __format__(self, format_spec):
    return str(self.resolve())
    
  def __str__(self):
    return str(self.resolve())

  def resolve(self):
    """
    See util function `resolve_quantified_variables`
    """
    resolved_exp = resolve_quantified_variables(self.original_expr, self.quantified_variableMap)
    return resolved_exp
  
  def is_completed(self):
    """
    Check whether all of quantified variables have value.
    True if all variables have value. Otherwise, False
    """
    return all(value is not None for value in self.quantified_variableMap.values())

class Prover9Proof:
  
  clauses: list[ProofClause]
  proof_steps: list[Prover9ProofStep]
  _proof_xml: str

  def __init__(self, proof_xml: str):
    self._proof = proof_xml
    tree = ET.parse(proof_xml)
    root = tree.getroot()
    number_of_proofs = root.attrib['number_of_proofs']
    proof_node = root.find('proof')
    clause_nodes = proof_node.findall('clause')
    self.clauses = [ProofClause(clause_node) for clause_node in clause_nodes]
    self.init()

  def init(self):
    implication_clauses = list(filter(lambda x: x.is_implication(), self.clauses))
    fact_clauses = list(filter(lambda x: x.is_fact(), self.clauses))
    proof_steps = [Prover9ProofStep(implication_clause) for implication_clause in implication_clauses]

    facts = []
    for fact_clause in fact_clauses:
      facts.extend(fact_clause.literals)
      
    for proof_step in proof_steps:
      applications = detect_application_expressions(proof_step.original_expr)
      for fact in facts:
        if isinstance(fact, ApplicationExpression):
          same_application = find_same_application(fact, applications)
          
          if same_application is not None:
            fact_args = fact.uncurry()[1]
            application_args = same_application.uncurry()[1]
            for fact_arg, application_arg in zip(fact_args, application_args):
              if isinstance(fact_arg, ConstantExpression):
                if isinstance(application_arg, AbstractVariableExpression):
                  proof_step.set_variable(application_arg.variable, fact_arg)  

        # Check for early finish if all quantified variables have value. 
        if proof_step.is_completed():
          break
      
      # TODO: Append resolved application_exp

class ProofClause:
  literals: list[Expression]
  attributes: Optional[list[str]]
  js_rule: str
  parents: Optional[list[int]]

  def __init__(self, clause_node: ET.Element):
    self.literals = [Expression.fromstring(literal_node.text.strip()) for literal_node in clause_node.findall('literal')] 
    self.attributes = [attribute_node.text.strip() for attribute_node in clause_node.findall('attribute')] 
    justification_node = clause_node.find('justification')
    j1_node = justification_node.find('j1')
    self.js_rule = j1_node.get('rule')
    parents_attr = j1_node.get('parents')
    if parents_attr:
      self.parents = parents_attr.strip().split(' ')
    else:
      self.parents = []
  
  def is_implication(self) -> bool:
    """
    Check where the clause is an assumed implication. Ex: (all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))) # label(non_clause).
    """
    return len(self.literals) == 1 and self.js_rule.lower() == 'assumption' and isinstance(self.literals[0], QuantifiedExpression)

  def is_fact(self) -> bool:
    """
    Check where the clause is an assumed fact. Ex: Takes(Tuan, NLP)
    """
    return self.js_rule.lower() == 'assumption' and len(self.literals) == 1 and isinstance(self.literals[0], ApplicationExpression)

