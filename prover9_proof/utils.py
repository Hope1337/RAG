from nltk.sem.logic import *

def detect_application_expressions(exp: Expression) -> list[ApplicationExpression]:
  """
  Get all application expression in the provided expression. 
  
  Ex: `all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))`  
  
  => { `Takes(x, y)`, `Passes(x, y)`, `AcquiresKnowledge(x)` }
  """
  applications = []

  if isinstance(exp, ApplicationExpression):
    applications.append(exp)
  
  if isinstance(exp, QuantifiedExpression):
    applications.extend(detect_application_expressions(exp.term))

  if isinstance(exp, BooleanExpression):
    applications.extend(detect_application_expressions(exp.first))
    applications.extend(detect_application_expressions(exp.second))

  if isinstance(exp, NegatedExpression):
    applications.extend(detect_application_expressions(exp.term))

  return applications

def detect_quantified_variables(exp: Expression) -> list[Variable]:
  """
  Get all quantified variables.

  Ex: `all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))`  
  
  => { `Variable('x')`, `Variable('y')` }
  """
  variables = []
  if isinstance(exp, QuantifiedExpression):
    variables.append(exp.variable)
    if (isinstance(exp.term, QuantifiedExpression)):
      variables.extend(detect_quantified_variables(exp.term))
  return variables

def resolve_quantified_variables(exp: Expression, variableMap: dict[Variable, ConstantExpression]) -> Expression:
  """
  Resolve quantified variables and return resolved expression.

  Example:
    exp: `all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))`  
    
    variableMap: 
    { 
      `Variable('x')`: `ConstantExpression('Tuan')`,
      `Variable('y')`: `ConstantExpression('NLP')`
    }
    
    => return: `Takes(Tuan, NLP) & Passes(Tuan, NLP) -> AcquiresKnowledge(Tuan)`
  """
  result = Expression.fromstring(str(exp))

  if not isinstance(result, QuantifiedExpression):
    return result
  
  # replace quantified variable
  quantified_variable = result.variable
  constant = variableMap.get(quantified_variable)
  
  if isinstance(result.term, QuantifiedExpression):
    result.term = resolve_quantified_variables(result.term, variableMap) 

  # replace quantified variable 
  if constant is not None:
    result = result.replace(quantified_variable, constant, replace_bound=True).term
  
  return result

def find_same_predicate(fact: ApplicationExpression, application_exps: list[ApplicationExpression]):
  """
  Find the same predicate. Return `None` if not found.

  Example:

  fact: `Takes(Tuan, NLP)`,

  application_exps: `Takes(x, y)`, `Takes(x, y, z)`, `Passes(x, y)`

  => Return `Takes(x, y)`
  """
  fact_function, fact_args = fact.uncurry()
  return next((application_exp for application_exp in application_exps if application_exp.uncurry()[0] == fact_function and len(application_exp.uncurry()[1]) == len(fact_args)), None)

def is_fact(exp: ApplicationExpression):
  """
  Check whether the expression is a fact. 
  
  Ex:
    - `Take(x, y)` -> False
    - `Take(Tuan, y)` -> False
    - `Take(Tuan, NLP)` -> True 
  """
  args = exp.uncurry()[1]
  return all(isinstance(arg, ConstantExpression) for arg in args)