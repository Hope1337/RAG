import torch
from knowledge.embeddings.RobertaEmbeddingGenerator import RobertaEmbeddingGenerator

from knowledge.ttp_parser import TtpParse

embedding_generator = RobertaEmbeddingGenerator(
    device=torch.device('cuda'),
    model_name='roberta-base',
    use_last_n_hidden_states=4,
)
parser = TtpParse(embedding_generator=embedding_generator)

sentence = 'Does "Natural Language Processing Semester 242" contain knowledge?.'
notation, alternative_logics, tensor_logics = parser.sentence_parse(sentence)

print('Penman notation ', notation)
print('\nalternative_logics ', alternative_logics)
print('\ntensor_logics: ')
for tensor_logic in tensor_logics:
    print('\t', str(tensor_logic))