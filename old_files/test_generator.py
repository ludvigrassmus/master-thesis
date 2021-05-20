import string
from nltk import word_tokenize
import csv
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
        
        for sentence in self.sent_corpus: 
            sent_words = word_tokenize(sentence)
            
            if keyword in sent_words:
                
                if type(self.sent_filter) == MCBertFilter: 
                    if self.sent_filter.approve(sentence, keyword, available_labels):
                        return sentence
                    # else:
                    #     print(sentence)
  
                # Type check doesn't work for BertFilter for some reason 
                else:
                    if self.sent_filter.approve(sentence, keyword):
                        return sentence
                    # else: 
                    #     print(sentence)
      
                    
        return None    
                
        
    # Generate a cloze test
    def generate_test(self, keyword, available_labels):   
        
        sentence = self.get_sentence(keyword, available_labels)
        
        # If no sentence is found, make this known. 
        if sentence == None:
            return 'ERROR: NO APPROVED SENTENCE FOUND'
        
        # Replace the keyword with an empty space
        tokenized_sentence = word_tokenize(sentence)
        
        detokenizer = TreebankWordDetokenizer()
        
        # Save separately to match both lower and upper case
        lowered = [word.lower() for word in tokenized_sentence]

        word_index = [i for i,word in enumerate(lowered) if word == keyword.lower()]
        
        for index in word_index:
            tokenized_sentence[index] = '____'
            
        return detokenizer.detokenize(tokenized_sentence)
        
    
    
    
    
    
    
    
