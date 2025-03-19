from knowledge.fol_premise_parser import PremiseFolParser
from knowledge.sentence_premise_parser import PremiseSentenceParser

# First order logic
fol_premises =[
    "∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))",
    "∀x (DrinkRegularly(x, coffee) ∨ (¬WantToBeAddictedTo(x, caffeine)))",
    "∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))",
    "¬(Student(rina) ⊕ ¬AwareThatDrug(rina, caffeine))",
    "¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))",
] 

fol_parser = PremiseFolParser()

parsed_fols = fol_parser.parse(fol_premises)
print(parsed_fols)

# sentences
sentence_premises = [
    "Every course contains knowledge.",
    "If a student takes a course and passes the exam, they acquire knowledge."
]

sentence_parser = PremiseSentenceParser()

parsed_sentences = sentence_parser.parse(fol_premises)
print(parsed_sentences)