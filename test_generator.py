import string
from nltk import word_tokenize, sent_tokenize
import csv
from bert_predictor import BertFilter
from MC_bert_predictor import MCBertFilter

class TestGenerator: 
    
    
    # predictor - a model to judge suitability of the sentences
    def __init__(self, sent_filter):
        self.sent_filter = sent_filter
        self.sent_corpus = []
        
        
      
    # Read corpus from file into data structure    
    def read_corpus(self, corpus_file_path, no_articles=20000):
        corpus = []
        with open('corpus.csv') as csvfile:
          reader = csv.reader(csvfile)
          for line in reader:
              corpus.append(line[0]) #.lower())
              
        self.read_sentences(corpus, no_articles)
        
        
    # Extract individual sentences from the corpus    
    def read_sentences(self, corpus, no_articles):    
        for article in corpus[:no_articles]:
            for sentence in sent_tokenize(article):
                if len(sentence) < 150:
                    self.sent_corpus.append(sentence)
        
              
        
    # Get an approved sentence from corpus
    def get_sentence(self, keyword, available_labels):        
        for sentence in self.sent_corpus: 
            sent_words = word_tokenize(sentence)
            if keyword in sent_words and len(sent_words) < 15:
                
                
                # if type(self.sent_filter) == BertFilter:  #THIS doesnt work for some reason
                #     if self.sent_filter.approve(sentence, keyword):
                #         return sentence
                    
                if type(self.sent_filter) == MCBertFilter: 
                    if self.sent_filter.approve(sentence, keyword, available_labels):
                        return sentence
                    else:
                        print(sentence)
  
                # The above type check doesn't work for BertFilter for some reason 
                else:
                    if self.sent_filter.approve(sentence, keyword):
                        return sentence
                    else: 
                        print(sentence)
  
                    
        return None    
                
        
    # Generate a cloze test 
    def generate_test(self, keyword, available_labels):   
        
        sentence = self.get_sentence(keyword, available_labels)
        
        if sentence == None:
            return 'ERROR: NO APPROVED SENTENCE FOUND'
        
        # Replace the keyword with an empty space
        tokenized_sentence = word_tokenize(sentence.lower())
        has_punctuation = tokenized_sentence[-1] in string.punctuation
    
        if tokenized_sentence.index(keyword) == 0:  # The keyword is the first in the sentence        
            return '___' + sentence[len(keyword):]        
    
        if has_punctuation and tokenized_sentence.index(keyword) == len(tokenized_sentence) - 2:
            return sentence[:-len(keyword)-1] + '___' + tokenized_sentence[-1]
        elif tokenized_sentence.index(keyword) == len(tokenized_sentence) - 1:
            return sentence[:-len(keyword)] + '___'
    
        keyword_env = ' ' + keyword + ' '   # Adding whitespace to avoid replacing where the string may be part of a longer word
       
        return sentence.replace(keyword_env, ' ___ ')
    
 
        
    
    
    
    
    
    
    