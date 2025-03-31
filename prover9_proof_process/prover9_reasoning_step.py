import re

from nltk.sem.logic import Expression, ApplicationExpression,  QuantifiedExpression

class Prover9ReasoningStep:
    index: int
    expression: Expression
    label: str
    rule_tag: str

    def __init__(self, reasoning_step):
        pattern = r"^(\d+)\s*(.*?)\.\s*(?:\[(.*?)\])?\.$"
        match = re.search(pattern, reasoning_step)
        if match:
            self.index = match.group(1).strip()
            rule = match.group(2).strip()
            self.label = match.group(3).strip() # assumption, goal, clausify(x), resolve(x, y,..), deny(x)
            rule_parts = rule.split('#')
            self.expression = Expression.fromstring(rule_parts[0])
            self.rule_tag = rule_parts[1].strip() if len(rule_parts) > 1 else None

    def is_fact(self) -> bool:
        """
        Check where the step is an assumed fact. Ex: Takes(Tuan, NLP)
        """
        return self.label.lower() == 'assumption' and isinstance(self.expression, ApplicationExpression)
    
    def is_original_rule(self) -> bool:
        """
        Check where the step is an assumed original rule. Ex: (all x all y (Takes(x,y) & Passes(x,y) -> AcquiresKnowledge(x))) # label(non_clause).
        """
        return self.label.lower() == 'assumption' and isinstance(self.expression, QuantifiedExpression) and self.rule_tag is not None
    
    def __str__(self):
        label = f'# {self.rule_tag})' if self.rule_tag is not None else ''
        return f'{self.index} {str(self.expression)} {label} [{self.label}]'