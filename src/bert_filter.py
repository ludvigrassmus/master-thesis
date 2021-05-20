import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
from nltk import word_tokenize


class BertFilter:
    
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        
     
     
    def decode(self, tokenizer, pred_data):
        pred_idx = pred_data.indices.tolist()
        pred_values = pred_data.values.tolist()
        top_clean = 3
        ignore_tokens = string.punctuation + '[PAD]'
        tokens = []
        for i in range(len(pred_idx)):
            idx = pred_idx[i]
            val = pred_values[i]
            token = ''.join(tokenizer.decode(idx).split())
            if token not in ignore_tokens:
                #tokens.append(token.replace('##', ''))
                tokens.append((token.replace('##', ''), val))
        return tokens[:top_clean]
    
    
    def encode(self, tokenizer, text_sentence, add_special_tokens=True):
        text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if tokenizer.mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'
        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
        return input_ids, mask_idx 
    
    
    def get_prediction(self, text_sentence):
        top_k = 10
        input_ids, mask_idx = self.encode(self.tokenizer, text_sentence)
        with torch.no_grad():
            predict = self.model(input_ids)[0]
        bert = self.decode(self.tokenizer, predict[0, mask_idx, :].topk(top_k))    
        return bert
        
        
    def mask_sentence(self, sentence: str, keyword: str) -> str:
        tokenized_sentence = word_tokenize(sentence.lower())
        ind = tokenized_sentence.index(keyword)
        tokenized_sentence[ind] = '<mask>'
        return ' '.join(tokenized_sentence)
    
    
    
    def approve(self, sentence: str, keyword: str, threshold: float) -> bool: 
        masked_sentence = self.mask_sentence(sentence, keyword)
        prediction = self.get_prediction(masked_sentence)
        words, values = list(zip(*prediction))
        #return keyword in prediction
        return (keyword in words) and (values[words.index(keyword)] > threshold)
    
    
    
        
    
    
    
    
    
    
        
