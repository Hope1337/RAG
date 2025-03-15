from nltk.sem.logic import (LogicParser, Expression)
from knowledge.utils import normalize_folText, convert_to_ttp_logic
import tensor_theorem_prover as ttp
from knowledge.embeddings.EmbeddingGenerator import EmbeddingGenerator
from transition_amr_parser.parse import AMRParser
import penman
from knowledge.AmrMergeGenerator import AmrMergeGenerator
from knowledge.amr_logic_to_tensor_logic import amr_logic_to_tensor_logic
from amr_logic_converter import AmrLogicConverter 

class TtpParse:
    
    nltk_logic_parser: LogicParser
    embedding_generator: EmbeddingGenerator
    amr_parser: AMRParser
    amr_merge_generator: AmrMergeGenerator
    amr_logic_converter: AmrLogicConverter

    def __init__(self, 
            embedding_generator: EmbeddingGenerator = None, 
            arm_parse_model: str = 'doc-sen-conll-amr-seed42',
            amr_merge_max_collapsed_per_node=5
        ):
        self.nltk_logic_parser = LogicParser()
        self.embedding_generator = embedding_generator
        self.amr_parser = AMRParser.from_pretrained(arm_parse_model)
        self.amr_merge_generator = AmrMergeGenerator(
            max_collapsed_per_node=amr_merge_max_collapsed_per_node
        )
        self.amr_logic_converter = AmrLogicConverter(
            use_implies_for_conditions=True,
            use_variables_for_instances=True,
        )
        pass

    def parse_FOL_to_ttp_logic(self, folText: str) -> ttp.Clause:
        try:
            nor_fol = normalize_folText(folText)
            exp = self.nltk_logic_parser.parse(nor_fol)
            storageVariables =[]
            return convert_to_ttp_logic(exp, storageVariables, self.embedding_generator)
        except Exception as e:
            SyntaxError(f"Cannot parse First Order Logic: {folText}")
        
    def sentence_parse(self, sentence):
        tokens, positions = self.amr_parser.tokenize(sentence)
        annotations, machines = self.amr_parser.parse_sentence(tokens)
        amr = machines.get_amr()
        penman_notation =  amr.to_penman(jamr=False, isi=True)
        tree = penman.parse(penman_notation)
        text = tree.metadata["tok"]
        alternative_trees = AmrMergeGenerator(
            max_collapsed_per_node=5
        ).generate_merged_amrs(tree)
        logic = self.amr_logic_converter.convert(tree)
        alternative_logics = [self.amr_logic_converter.convert(tree) for tree in alternative_trees]
        embeddings= None
        if self.embedding_generator is not None:
            [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        tensor_logics = [amr_logic_to_tensor_logic(logic, embeddings) for logic in alternative_logics]
        return penman_notation, logic, alternative_logics, tensor_logics
    
    def sentence_parse_no_merge(self, sentence):
        tokens, positions = self.amr_parser.tokenize(sentence)
        annotations, machines = self.amr_parser.parse_sentence(tokens)
        amr = machines.get_amr()
        penman_notation =  amr.to_penman(jamr=False, isi=True)
        tree = penman.parse(penman_notation)
        text = tree.metadata["tok"]
        logic = self.amr_logic_converter.convert(tree)
        embeddings= None
        if self.embedding_generator is not None:
            [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        tensor_logic = amr_logic_to_tensor_logic(logic, embeddings)
        return penman_notation, logic, tensor_logic
