from concurrent import futures
import grpc
import os
import random
import numpy as np
from nltk import word_tokenize
import torch
import logging
import test_generator_pb2
import test_generator_pb2_grpc
import csv
import math
from nltk.tokenize.treebank import TreebankWordDetokenizer


from bert_filter import BertFilter
from cefr_word_scorer import CefrWordScorer


CEFR_LEVELS = {'A1': ('A1', 'A2', 'A2+'), 'A2': ('A2', 'A1', 'A2+', 'B1'),
               'A2+': ('A2+', 'A2', 'B1', 'B1+'), 'B1': ('B1', 'A2+', 'B1+', 'B2'),
               'B1+': ('B1+', 'B1', 'A2+', 'B2'), 'B2': ('B2', 'B1+', 'B1', 'B2+'),
               'B2+': ('B2+', 'B2', 'B1+', 'B1'), 'C1': ('C1', 'B2+', 'B2', 'C2'),
               'C2': ('C2', 'C1', 'B2+', 'B2')}



def read_corpus(corpus_dict: str):
    sent_corpora = {}
    for filename in os.listdir(corpus_dict):
        if not filename.startswith('.'):
            sents = []
            with open(corpus_dict + '/' + filename) as file:
                reader = csv.reader(file)
                for line in reader:
                    sents.append(line[0])
            sent_corpora[filename.split('_')[0]] = sents
    return sent_corpora



class TestGeneratorServicer(test_generator_pb2_grpc.TestGeneratorServicer):

    def __init__(self):
        print('Loading BertFilter...')
        self.sent_filter = BertFilter()
        print('Loading sentences...')
        self.sent_corpora = read_corpus('CEFR_sentences')
        print('Loading Word Scorer...')
        self.word_scorer = CefrWordScorer('word_spacy_pos_to_gse_map.pickle')
        print("Complete!")


    def GetSentence(self, request, context):
        word = request.text
        cefr = self.word_scorer.get_score(word)
        cloze_test = self.generate_test(word, cefr, 10.0)

        cloze_message = test_generator_pb2.Sentence(text=cloze_test)

        return cloze_message


    ## Helper functions for the TestGenerator

    # Returns a group of approved sentences from the corpus
    def get_sent_candidates(self, keyword: str, cefr: str, threshold: float) -> [str]:

        sent_candidates = []

        shuffled_sentences = self.sent_corpora[cefr]
        random.shuffle(shuffled_sentences)

        for sentence in shuffled_sentences:

            sent_words = word_tokenize(sentence)

            if keyword in sent_words:

                if self.sent_filter.approve(sentence, keyword, threshold):
                    sent_candidates.append(sentence)

            if len(sent_candidates) == 10:
                break

        return sent_candidates



    # Returns the probability score of the sentence
    def get_score(self, sentence: str) -> float:
        tokenizer = self.sent_filter.tokenizer

        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = self.sent_filter.model(tensor_input)[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data

        return math.exp(loss)


    # Generate a cloze test for given keyword
    def generate_test(self, keyword: str, cefr: str, threshold: float) -> str:

        # Get the allowed CEFR levels to check through
        cefr_levels = CEFR_LEVELS[cefr]

        for cefr_level in cefr_levels:
            sent_candidates = self.get_sent_candidates(keyword, cefr_level, threshold)
            if sent_candidates:
                break

        # If no sentence is found
        if not sent_candidates:
            return 'ERROR: NO SENTENCE FOUND'

        # Find the best candidate
        sent_scores = [self.get_score(sent) for sent in sent_candidates]
        sentence = sent_candidates[np.argmax(sent_scores)]


        # Replace the keyword with an empty space
        tokenized_sentence = word_tokenize(sentence)

        detokenizer = TreebankWordDetokenizer()

        # Save separately to match both lower and upper case and return sentence with cases intact
        lowered = [word.lower() for word in tokenized_sentence]

        word_index = [i for i,word in enumerate(lowered) if word == keyword.lower()]

        for index in word_index:
            tokenized_sentence[index] = '____'

        return detokenizer.detokenize(tokenized_sentence)




def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_generator_pb2_grpc.add_TestGeneratorServicer_to_server(
        TestGeneratorServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()

