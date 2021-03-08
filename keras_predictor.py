import keras 
import numpy as np
import tensorflow as tf
from nltk import word_tokenize



class KerasFilter: 
    
    
    def __init__(self):
        tf.tf_xla_enable_xla_devices = True
        self.model = keras.models.load_model("preposition_predictor")
        self.w_size = 4
        
        self.word2idx = np.load('word2idx.npy', allow_pickle='TRUE').item()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        self.label2idx = np.load('label2idx.npy', allow_pickle='TRUE').item()
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        
        

    
    def approve(self, sentence, keyword, threshold=0.5):  # Check if keyword is predictable in sentence
        # Tokenize sentence and set to lowercase
        tokenized_sentence = word_tokenize(sentence.lower())   
        # Pad sentence on both ends 
        tokenized_sentence = ['<PAD>']*self.w_size + tokenized_sentence + ['<PAD>']*self.w_size  
        # Get index of the keyword  
        keyword_idx = tokenized_sentence.index(keyword) 
        # Get the context of the keyword   
        model_input_words = tokenized_sentence[keyword_idx-self.w_size: keyword_idx] + tokenized_sentence[keyword_idx+1: keyword_idx+self.w_size+1]  
        # Get the word indices of the context words
        model_input_idxs = [self.word2idx[w] for w in model_input_words]  
        # Try to predict the keyword  
        pred = self.model.predict([model_input_idxs])
        prediction = self.idx2label[np.argmax(pred)]   
        prediction_conf = max(pred[0]) 
        # Return True only if the keyword can be predicted confidently enough
        return (keyword==prediction) and (prediction_conf > threshold)













