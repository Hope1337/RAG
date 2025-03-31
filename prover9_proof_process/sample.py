from sample_proof import _proof
import re
from prover9_reasoning_step import Prover9ReasoningStep
from prover9_proof import Prover9ProofStep 
from nltk.sem.logic import Variable, ConstantExpression

def detect_section(section_prefix, section_suffix, text):
    pattern = re.search(f"=+ {section_prefix} =+\n(.*?)=+ {section_suffix} =+", text, re.DOTALL)
    return pattern.group(1).strip() if pattern else None

proof_section = detect_section('PROOF', 'end of proof', _proof)

reasoning = proof_section.strip().split("\n")

# Tìm các reasoning steps với dòng bắt đầu bằng số
reasoningSteps = [Prover9ReasoningStep(step) for step in reasoning if re.match(r"^\d+", step)]

proof_step = Prover9ProofStep(reasoningSteps[0])
proof_step.set_variable(Variable('x'), ConstantExpression(Variable('Tuan')))
proof_step.set_variable(Variable('y'), ConstantExpression(Variable('NLP')))
print(proof_step.resolve())