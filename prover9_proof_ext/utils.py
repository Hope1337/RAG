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
    applications.append(detect_application_expressions(exp.term))

  if isinstance(exp, BooleanExpression):
    applications.append(detect_application_expressions(exp.first))
    applications.append(detect_application_expressions(exp.second))

  if isinstance(exp, NegatedExpression):
    applications.append(detect_application_expressions(exp.term))

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

  if not isinstance(exp, QuantifiedExpression):
    return exp
  
  quantified_variable = exp.variable
  constant = variableMap[quantified_variable]

  if constant is None:
    return exp
  replaced_exp = exp.replace(quantified_variable, constant, replace_bound= True)
  return resolve_quantified_variables(replaced_exp.term, variableMap)

def find_same_application(fact: ApplicationExpression, application_exps: list[ApplicationExpression]):
  """
  Find the same application expression with provided fact. Return `None` if not found.

  Example:

  fact: `Takes(Tuan, NLP)`,

  application_exps: `Takes(x, y)`, `Takes(x, y, z)`, `Passes(x, y)`

  => Return `Takes(x, y)`
  """
  fact_function, fact_args = fact.expression.uncurry()
  return next((application_exp for application_exp in application_exps if application_exp.uncurry()[0] == fact_function & len(application_exp.uncurry()[1]) == len(fact_args)), None)

def detect_section(section_prefix, section_suffix, text):
    pattern = re.search(f"=+ {section_prefix} =+\n(.*?)=+ {section_suffix} =+", text, re.DOTALL)
    return pattern.group(1).strip() if pattern else None