from ..utils.data import DatasetHandler
from .nmt import EngTokenizer


class QuestionAnswer(DatasetHandler):

    def __init__(self, questions, answers, num_instances=10000):
        encoder_tokenizer = EngTokenizer(questions[:num_instances], is_encoder=True)
        decoder_tokenizer = EngTokenizer(answers[:num_instances])
        super().__init__(encoder_tokenizer, decoder_tokenizer)
