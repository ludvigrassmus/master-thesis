from nltk import word_tokenize
import csv
import torch
import math
import os
import numpy as np
from bert_predictor import BertFilter
from MC_bert_predictor import MCBertFilter
from nltk.tokenize.treebank import TreebankWordDetokenizer


'''
The CEFR_LEVELS dictionary represents which CEFR level corpora we are allowing
to check in for which CEFR word level. So if a word is classified as CEFR level 
A2, we first check for sentences in the A2 level corpus, then A1, and lastly
A2+. 
'''

CEFR_LEVELS = {'A1': ('A1', 'A2', 'A2+'), 'A2': ('A2', 'A1', 'A2+'), 
               'A2+': ('A2+', 'A2', 'B1', 'B1+'), 'B1': ('B1', 'A2+', 'B1+', 'B2'), 
               'B1+': ('B1+', 'B1', 'A2+', 'B2'), 'B2': ('B2', 'B1+', 'B1', 'B2+'), 
               'B2+': ('B2+', 'B2', 'B1+', 'B1'), 'C1': ('C1', 'B2+', 'B2', 'C2'), 
               'C2': ('C2', 'C1', 'B2+', 'B2')}

class TestGenerator: 
    
    # predictor - a model to judge suitability of the sentences
    def __init__(self, sent_filter):
        self.sent_filter = sent_filter
        #self.sent_corpus = []   
        #self.read_corpus(corpus_file)
        self.sent_corpora = {}
        
    
    def read_corpus(self, corpus_dict: str):
        # with open(corpus_file) as csvfile:
        #     reader = csv.reader(csvfile)
        #     for line in reader:
        #         self.sent_corpus.append(line[0])
        for filename in os.listdir(corpus_dict):
            if not filename.startswith('.'):
                sents = []
                with open(corpus_dict + '/' + filename) as file:
                    reader = csv.reader(file)
                    for line in reader:
                        sents.append(line[0])
                self.sent_corpora[filename.split('_')[0]] = sents
              
        
    # Get an approved sentence from corpus
    def get_sentence(self, keyword: str, available_labels: [str], cefr: str) -> [str]:  
        
        sent_candidates = []
        
        for sentence in self.sent_corpora[cefr]: 
            
            sent_words = word_tokenize(sentence)
            
            if keyword in sent_words:
                
                if type(self.sent_filter) == MCBertFilter: 
                    if self.sent_filter.approve(sentence, keyword, available_labels):
                        sent_candidates.append(sentence)
  
                # Type check doesn't work for BertFilter for some reason 
                else:
                    if self.sent_filter.approve(sentence, keyword):
                        sent_candidates.append(sentence)
                        
            if len(sent_candidates) == 10:
                break
                    
        return sent_candidates    
                
    
    def get_score(self, sentence: str) -> float:
        tokenizer = self.sent_filter.tokenizer
        
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = self.sent_filter.model(tensor_input)[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
        
        return math.exp(loss)
    
        
    # Generate a cloze test for given keyword
    def generate_test(self, keyword: str, available_labels: [str], cefr: str) -> str:
        

        cefr_levels = CEFR_LEVELS[cefr]

        
        for cefr_level in cefr_levels:    
            sent_candidates = self.get_sentence(keyword, available_labels, cefr_level)
            if sent_candidates:
                break

        # If no sentence is found
        if not sent_candidates:
            return 'ERROR: NO SENTENCE FOUND'
        
        # Find best candidate
        sent_scores = [self.get_score(sent) for sent in sent_candidates]
        sentence = sent_candidates[np.argmax(sent_scores)]

        
        # Replace the keyword with an empty space
        tokenized_sentence = word_tokenize(sentence)
        
        detokenizer = TreebankWordDetokenizer()
        
        # Save separately to match both lower and upper case and return correctly
        lowered = [word.lower() for word in tokenized_sentence]

        word_index = [i for i,word in enumerate(lowered) if word == keyword.lower()]
        
        for index in word_index:
            tokenized_sentence[index] = '____'
            
        return detokenizer.detokenize(tokenized_sentence)
        
    
    
    
    
    
    
    