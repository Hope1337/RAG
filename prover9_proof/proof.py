from __future__ import annotations
from nltk.sem.logic import Expression, Variable, ConstantExpression, QuantifiedExpression, ApplicationExpression, AbstractVariableExpression
import xml.etree.ElementTree as ET
from typing import Optional

from prover9_proof.utils import detect_application_expressions, detect_quantified_variables, resolve_quantified_variables, find_same_predicate, is_fact

class ProofStep:
  """
  Represent for a step of proof. Ex: Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan)
  """

  premise: Expression 
  """
  Reference premise.

  Ex: all x y. ( Takes(x, y) & Passes(x, y) -> AcquiresKnowledge(x) )
  """

  quantified_variableMap: dict[Variable, ConstantExpression] # 
  """
  Contain value of quantified variables. 

  Ex: {x -> Tuan, y -> NLP}
  """
  
  progression: Expression
  """
  The expression result of unifying the quantified variables from `quantified_variableMap` into `premise`. 

  Ex: Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan) 
  """

  def __init__(self, rule_clause: ProofClause):
    assert rule_clause.is_rule() == True

    self.premise = rule_clause.literals[0]
    self.progression = rule_clause.literals[0]
    quantified_variables = detect_quantified_variables(self.premise)
    self.quantified_variableMap = { variable: None for variable in quantified_variables} # initial None for all quantified variables
  
  def set_variable(self, variable: Variable, constant: ConstantExpression):
    self.quantified_variableMap[variable] = constant
    self.progression = resolve_quantified_variables(self.premise, self.quantified_variableMap)
  
  def __str__(self):
    return str(self.progression)
  
  def is_completed(self):
    return not isinstance(self.progression, QuantifiedExpression)

class Proof:
  
  clauses: list[ProofClause]
  proof_steps: list[ProofStep]
  _proof_node: ET.Element

  def __init__(self, proof_node: ET.Element):
    self._proof_node = proof_node
    clause_nodes = proof_node.findall('clause')
    self.clauses = [ProofClause(clause_node) for clause_node in clause_nodes]
    self.init()

  def init(self):
    rule_clauses = list(filter(lambda x: x.is_rule(), self.clauses))
    fact_clauses = list(filter(lambda x: x.is_fact(), self.clauses))
    proof_steps = [ProofStep(rule_clause) for rule_clause in rule_clauses]

    facts = []
    for fact_clause in fact_clauses:
      facts.extend(fact_clause.literals)
      
    for proof_step in proof_steps:
      applications = detect_application_expressions(proof_step.premise)
      for fact in facts:
        if isinstance(fact, ApplicationExpression):
          same_application = find_same_predicate(fact, applications)
          
          if same_application is not None:
            fact_args = fact.uncurry()[1]
            application_args = same_application.uncurry()[1]
            for fact_arg, application_arg in zip(fact_args, application_args):
              if isinstance(fact_arg, ConstantExpression):
                if isinstance(application_arg, AbstractVariableExpression):
                  # Unifying
                  proof_step.set_variable(application_arg.variable, fact_arg)

                  # Append resolved application_exp
                  extend_facts = [exp for exp in detect_application_expressions(proof_step.progression) if is_fact(exp) and exp not in facts]
                  facts.extend(extend_facts)

        # early finish if all quantified variables have value. 
        if proof_step.is_completed():
          break
    self.proof_steps = proof_steps

  def __str__(self):
    text = ''
    for i in range(len(self.proof_steps)):
      text += '\nStep {}:'.format(i)
      text += '\n\tPremise: {}'.format(self.proof_steps[i].premise)
      text += '\n\t=> {}'.format(self.proof_steps[i].progression)
    return text
  
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
  
  def is_rule(self) -> bool:
    """
    Check where the clause is an assumed rule. Ex: (all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))) # label(non_clause).
    """
    return len(self.literals) == 1 and self.js_rule.lower() == 'assumption' and isinstance(self.literals[0], QuantifiedExpression)

  def is_fact(self) -> bool:
    """
    Check where the clause is an assumed fact. Ex: Takes(Tuan, NLP)
    """
    return self.js_rule.lower() == 'assumption' and len(self.literals) == 1 and isinstance(self.literals[0], ApplicationExpression)

