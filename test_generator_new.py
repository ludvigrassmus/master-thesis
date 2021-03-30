from nltk import word_tokenize
import csv
import torch
import math
import numpy as np
from bert_predictor import BertFilter
from MC_bert_predictor import MCBertFilter
from nltk.tokenize.treebank import TreebankWordDetokenizer


class TestGenerator: 
    
    
    # predictor - a model to judge suitability of the sentences
    def __init__(self, sent_filter, corpus_file):
        self.sent_filter = sent_filter
        self.sent_corpus = []   
        self.read_corpus(corpus_file)


        
    
    def read_corpus(self, corpus_file):
        with open(corpus_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.sent_corpus.append(line[0])
              
        
    # Get an approved sentence from corpus
    def get_sentence(self, keyword, available_labels):  
        
        sent_candidates = []
        
        for sentence in self.sent_corpus: 
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
                
    
    def get_score(self, sentence):
        tokenizer = self.sent_filter.tokenizer
        
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = self.sent_filter.model(tensor_input)[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
        return math.exp(loss)
    
        
    # Generate a cloze test
    def generate_test(self, keyword, available_labels):   
        
        sent_candidates = self.get_sentence(keyword, available_labels)
        
        # If no sentence is found, make this known. 
        if not sent_candidates:
            return 'ERROR: NO APPROVED SENTENCE FOUND'
        
        # Find best candidate
        sent_scores = [self.get_score(sent) for sent in sent_candidates]
        sentence = sent_candidates[np.argmax(sent_scores)]

        
        # Replace the keyword with an empty space
        tokenized_sentence = word_tokenize(sentence)
        
        detokenizer = TreebankWordDetokenizer()
        
        # Save separately to match both lower and upper case
        lowered = [word.lower() for word in tokenized_sentence]

        word_index = [i for i,word in enumerate(lowered) if word == keyword.lower()]
        
        for index in word_index:
            tokenized_sentence[index] = '____'
            
        return detokenizer.detokenize(tokenized_sentence)
        
    
    
    
    
    
    
    