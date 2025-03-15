from knowledge.ttp_parser import TtpParse
from tensor_theorem_prover import ResolutionProver

# Parse premises to knowledge
premises =[
    "∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))",
    "∀x (DrinkRegularly(x, coffee) ∨ (¬WantToBeAddictedTo(x, caffeine)))",
    "∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))",
    "¬(Student(rina) ⊕ ¬AwareThatDrug(rina, caffeine))",
    "¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))",
] 
embeddingGenerator = None
parser = TtpParse(embedding_generator=embeddingGenerator)
knowledge = [parser.parse_FOL_to_ttp_logic(premise) for premise in premises]

print('knowledge:')
for rule in knowledge:
    print('\t', rule)

# Parse conclusion
conclusion = "¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))"
goal = parser.parse_FOL_to_ttp_logic(conclusion)

prover = ResolutionProver(knowledge=knowledge)
print('Goal prove:', goal)
proofs = prover.prove_all(goal=goal)
print(f'Found {len(proofs)} proofs')
if len(proofs) > 0:
    print(proofs[0])