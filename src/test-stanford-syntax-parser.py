import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from knowledge.ttp_parser import TtpParse

nltk.download('stopwords')
StanfordCoreNLP_path = 'D:/Code/ARM_Codes/RAG/src/knowledge/stanford-corenlp-4.5.8'
stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None

sentence = "Will a student gain more knowledge when they have understanding?"

tokens = en_model.word_tokenize(sentence)
tokens_tagged = en_model.pos_tag(sentence)

GRAMMAR = """  
        NP: {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)
        VP: {<VB.*><NP|PP|CLAUSE>*} # Động từ + Cụm danh từ, giới từ hoặc mệnh đề (tùy chọn)
        """

np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
# keyphrase_candidate = []
np_pos_tag_tokens = np_parser.parse(tokens_tagged)

print('np_pos_tag_tokens', np_pos_tag_tokens)
# (S
#   Will/MD
#   a/DT
#   (NP student/NN)
#   (VP gain/VB)
#   (# AP more/JJR)
#   (NP knowledge/NN)
#   when/WRB
#   they/PRP
#   (VP have/VBP (NP understanding/NN))
#   ?/.)

ttp_parser = TtpParse(embedding_generator=None)

notation, logic, alternative_logics, tensor_logics = ttp_parser.sentence_parse(sentence)
print('penman', notation)

# (g / gain-02~3
#     :ARG0 (p / person~2
#         :ARG0-of (s / study-01~2))
#     :ARG1 (k / know-01~5
#         :ARG0 p
#         :mod (m / more~4))
#     :polarity (a / amr-unknown~10)
#     :time (u / understand-01~9
#         :ARG0 p))