from __future__ import annotations
from dataclasses import dataclass
from nltk.sem.logic import Expression, Variable, ConstantExpression, QuantifiedExpression, ApplicationExpression, BooleanExpression, NegatedExpression, IndividualVariableExpression, AbstractVariableExpression
from prover9_reasoning_step import Prover9ReasoningStep

class Prover9ProofStep:
  """
  Class for an proof step. Ex: Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan)
  """
  expression: Expression # Ex: all x y. ( Takes(x, y) & Passes(x, y) -> AcquiresKnowledge(x) )
  quantifiedVariableMap: dict[Variable, ConstantExpression] # Ex: {x -> Tuan, y -> NLP}
  # reference: int # id of referenced premise

  def __init__(self, reasoningStep: Prover9ReasoningStep):
    self.expression = reasoningStep.expression
    quantifiedVariables = []
    detect_quantified_variables(self.expression, quantifiedVariables)
    self.quantifiedVariableMap = { quantifiedVariable: None for quantifiedVariable in quantifiedVariables} # initial None for all quantified variables
  
  def set_variable(self, variable: Variable, constant: ConstantExpression):
    self.quantifiedVariableMap[variable] = constant

  def resolve(self):
    """
    Replace all variable map into expression and return replaced expression.

    Ex:  Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan)
    """
    resolved_exp = replace_variables(self.expression, self.quantifiedVariableMap)
    return resolved_exp
  
  def is_resolved_variables(self):
    """
    Check whether all of variables are replaced with a constant.
    True if all variables are replaced. Otherwise, False
    """
    return all(value is not None for value in self.quantifiedVariableMap.values())

class Prover9Proof:
  
  steps: list[Prover9ProofStep]

  def __init__(self, steps: list[Prover9ProofStep]):
    self.steps = steps

  def parse(reasoningSteps: list[Prover9ReasoningStep])-> "Prover9Proof":
    rule_reasoning_steps = list(filter(lambda x: x.is_original_rule(), reasoningSteps))
    fact_reasoning_steps = list(filter(lambda x: x.is_fact(), reasoningSteps))
    proof_steps = [Prover9ProofStep(applied_rule) for applied_rule in rule_reasoning_steps]

    facts = [fact_reasoning_step.expression for fact_reasoning_step in fact_reasoning_steps]
    for proof_step in proof_steps:
      application_exps = detect_application_exprs(proof_step.expression)
      for fact in facts:
        if isinstance(fact, ApplicationExpression):
          mapped_application_exp = find_mapped_application_exps(fact, application_exps)
          
          if mapped_application_exp is not None:
            fact_args = fact.uncurry()[1]
            mapped_application_exp_args = mapped_application_exp.uncurry()[1]
            for fact_arg, mapped_application_exp_arg in zip(fact_args, mapped_application_exp_args):
              if isinstance(fact_arg, ConstantExpression):
                if isinstance(mapped_application_exp_arg, AbstractVariableExpression):
                  proof_step.set_variable(mapped_application_exp_arg.variable, fact_arg)  

        # Check for early finish 
        if proof_step.is_resolved_variables():
          break
      
      # TODO: Append resolved application_exp

    return Prover9Proof(proof_steps)

def find_mapped_application_exps(fact: ApplicationExpression, application_exps: list[ApplicationExpression]):
  fact_function, fact_args = fact.expression.uncurry()
  return next((application_exp for application_exp in application_exps if application_exp.uncurry()[0] == fact_function & len(application_exp.uncurry()[1]) == len(fact_args)), None)

def detect_quantified_variables(exp: Expression, outQuantifiedVariables: list[Variable]):
  """
  Get all quantified variables of the expression and append to param `outQuantifiedVariables`
  """
  if isinstance(exp, QuantifiedExpression):
    outQuantifiedVariables.append(exp.variable)
    if (isinstance(exp.term, QuantifiedExpression)):
      detect_quantified_variables(exp.term, outQuantifiedVariables)

def replace_variables(exp: Expression, variableMap: dict[Variable, ConstantExpression]) -> Expression:
  if not isinstance(exp, QuantifiedExpression):
    return exp
  
  quantified_variable = exp.variable

  constant = variableMap[quantified_variable]
  if constant is None:
    return ValueError('Missing value for ' + str(quantified_variable))

  replaced_exp = exp.replace(quantified_variable, constant, replace_bound= True)
  return replace_variables(replaced_exp.term, variableMap)

def detect_application_exprs(exp: Expression) -> list[ApplicationExpression]:
  """
  Get all application expression in the provided expression. Ex. Takes(x, y), ...
  """
  application_exps = []

  # Nếu expr là ApplicationExpression. Ex: Takes(x, y)
  if isinstance(exp, ApplicationExpression):
    application_exps.append(exp)
  
  if isinstance(exp, QuantifiedExpression):
    application_exps.append(detect_application_exprs(exp.term))

  if isinstance(exp, BooleanExpression):
    application_exps.append(detect_application_exprs(exp.first))
    application_exps.append(detect_application_exprs(exp.second))

  if isinstance(exp, NegatedExpression):
    application_exps.append(detect_application_exprs(exp.term))