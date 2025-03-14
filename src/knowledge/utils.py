from nltk.sem.logic import (
    LogicParser, 
    AllExpression, ExistsExpression, 
    ImpExpression, 
    AndExpression, OrExpression, NegatedExpression,
    ApplicationExpression,
    Expression,
    ConstantExpression
    )
import tensor_theorem_prover as ttp
from knowledge.embeddings.EmbeddingGenerator import EmbeddingGenerator, WordEmbeddings
import re

from penman.tree import Node, Tree

symbol_replacements = {
    'exists ': ['∃'],
    'forall ': ['∀'],
    '& ': ['∧'],
    '| ': ['∨'],
    '! ': ['¬'],
    ' => ': ['→']
}

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

def convert_to_ttp_logic(exp: Expression, storageVariables: list[ttp.Variable], embeddingGenerator: EmbeddingGenerator = None) -> ttp.Clause: 
    if isinstance(exp, AllExpression):
        ttp_variable = ttp.Variable(exp.variable.name)
        storageVariables.append(ttp_variable)
        return ttp.All(
            variable = ttp_variable,
            body= convert_to_ttp_logic(exp.term, storageVariables, embeddingGenerator)
        )
    elif isinstance(exp, ExistsExpression):
        ttp_variable = ttp.Variable(exp.variable.name)
        storageVariables.append(ttp_variable)
        return ttp.Exists(
            variable = ttp_variable,
            body= convert_to_ttp_logic(exp.term, storageVariables, embeddingGenerator)
        )
    elif isinstance(exp, ImpExpression):
        return ttp.Implies(
            antecedent= convert_to_ttp_logic(exp.first, storageVariables, embeddingGenerator),
            consequent= convert_to_ttp_logic(exp.second, storageVariables, embeddingGenerator),
        )
    elif isinstance(exp, AndExpression):
        return ttp.And(
            convert_to_ttp_logic(exp.first, storageVariables, embeddingGenerator),
            convert_to_ttp_logic(exp.second, storageVariables, embeddingGenerator),
        )
    elif isinstance(exp, OrExpression):
        return ttp.Or(
            convert_to_ttp_logic(exp.first, storageVariables, embeddingGenerator),
            convert_to_ttp_logic(exp.second, storageVariables, embeddingGenerator),
        )
    elif isinstance(exp, NegatedExpression):
        return ttp.Not(
            convert_to_ttp_logic(exp.term, storageVariables, embeddingGenerator),
        )
    elif isinstance(exp, ApplicationExpression):
        predicate_exp, args = exp.uncurry()
        assert isinstance(predicate_exp, ConstantExpression)
        predicate = predicate_exp.variable
        terms = []
        for arg in args:
            if isinstance(arg, ConstantExpression):
                terms.append(ttp.Constant(
                    symbol= arg.variable.name,
                    embedding= embeddingVariable(embeddingGenerator, arg.variable.name)
                ))
            else:
                variable_name = arg.variable.name
                existing_variables = list(filter(lambda x: x.name == variable_name, storageVariables))
                if len(existing_variables) > 0 :
                    terms.append(existing_variables[0])
                else:
                    terms.append(ttp.Variable(variable_name))

        return ttp.Atom(
            predicate= ttp.Predicate(
                symbol=predicate.name,
                embedding= embeddingVariable(embeddingGenerator, predicate.name)
            ),
            terms= terms
        )


def embeddingVariable(embeddingGenerator: EmbeddingGenerator, camelCaseVariableName: str) -> WordEmbeddings:
    if embeddingGenerator is None :
        return None
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', camelCaseVariableName)
    return embeddingGenerator.generate_word_embeddings([" ".join(words)])[0]

def count_nodes_in_amr_tree(tree: Tree) -> int:
    return _count_nodes_in_amr_tree_inner(tree.node)

def _count_nodes_in_amr_tree_inner(node: Node) -> int:
    num_nodes = 1
    _instance, instance_info = node
    _predicate, *edges = instance_info
    for _role, target in edges:
        if isinstance(target, tuple):
            num_nodes += _count_nodes_in_amr_tree_inner(target)
        else:
            # if the child isn't a node, treat this depth as 2 (cur node + child)
            num_nodes += 1
    return num_nodes

