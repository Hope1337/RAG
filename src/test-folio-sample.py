from knowledge.ttp_parser import TtpParse
import torch
from knowledge.embeddings.RobertaEmbeddingGenerator import RobertaEmbeddingGenerator
from tensor_theorem_prover import ResolutionProver, Implies

embedding_generator = RobertaEmbeddingGenerator(
    device=torch.device('cuda'),
    model_name='roberta-base',
    use_last_n_hidden_states=4,
)

parser = TtpParse(embedding_generator=embedding_generator)

question = "Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug."
# ¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))

# Parse premises to knowledge
premises =[
    "∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))",
    "∀x (DrinkRegularly(x, coffee) ∨ (¬WantToBeAddictedTo(x, caffeine)))",
    "∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))",
    "¬(Student(rina) ⊕ ¬AwareThatDrug(rina, caffeine))",
    "¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))",
] 
knowledge = [parser.parse_FOL_to_ttp_logic(premise) for premise in premises]

print('knowledge:')
for rule in knowledge:
    print('\t', rule)

# notation, logic, alternative_logics, tensor_logics = parser.sentence_parse(question)
# print('alternative_logics:', alternative_logics)
# for alternative_logic in alternative_logics:
#     print('\t', str(alternative_logic))
notation, logic, tensor_logic = parser.sentence_parse_no_merge(question)

print('logic:', logic)
print('tensor_logic: ', tensor_logic)

goal = parser.parse_FOL_to_ttp_logic("¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))")
goal = parser.parse_FOL_to_ttp_logic("(¬AwareThatDrug(rina, caffeine))")
# goal = tensor_logic
prover = ResolutionProver(knowledge=knowledge)
print('Goal prove:', goal)
proof = prover.prove(goal=goal)
print(proof)

# premises1 = [
#     "All people who regularly drink coffee are dependent on caffeine.",
#     "People regularly drink coffee, or they don't want to be addicted to caffeine, or both.",
#     "No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.",
#     "Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug. ",
#     "Rina is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine."
# ]

# knowledge1 = [parser.sentence_parse_no_merge(premise)[2] for premise in premises1]

# goal1 = parser.sentence_parse_no_merge(question)[2]

# prover1 = ResolutionProver(knowledge=knowledge1)
# proof1 = prover1.prove(goal=goal1)