import pickle 
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_gse_score(scores: List[float], scoring_type: str):
    if scoring_type == 'mean':
        return np.mean(scores)
    elif scoring_type == 'min':
        return np.min(scores)
    elif scoring_type == 'max':
        return np.max(scores)
    elif scoring_type == 'median':
        return np.median(scores)
    else:
        raise KeyError(f"Unable to process with scoring type {scoring_type}")

def get_gse_map(file_path: str) -> Dict[Tuple[str, str], List[int]]:
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def gse_to_cefr9(score: float) -> float:
    if score <= 30:
        return score / 8 - 3.25
    elif score <= 36:
        return score / 6 - 4.5
    elif score <= 43:
        return score / 7 - 51 / 14
    elif score <= 67:
        return score / 8 - 2.875
    elif score <= 85:
        return score / 9 - 35 / 18
    else:
        return score / 5 - 9.5
    
    
def num2cefr(score: float) -> str:
    if score < 0.5:
        return 'A1'
    elif 0.5 <= score < 1.5:
        return 'A2'
    elif 1.5 <= score < 2.5:
        return 'A2+'
    elif 2.5 <= score < 3.5:
        return 'B1'
    elif 3.5 <= score < 4.5:
        return 'B1+'
    elif 4.5 <= score < 5.5:
        return 'B2'
    elif 5.5 <= score < 6.5:
        return 'B2+'
    elif 6.5 <= score < 7.5:
        return 'C1'
    elif score > 7.5:
        return 'C2'
    else:
        return 'Invalid score'
    
    
def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.VERB

def get_lemma(phrase: str) -> str:
    pos = pos_tag([phrase])[0][1]
    lemma = lemmatizer.lemmatize(phrase, get_wordnet_pos(pos))
    return lemma


class CefrWordScorer:
    
    def __init__(self, gse_score_path):
        self.gse_map = get_gse_map(gse_score_path)
        self.gse_map_no_pos = self.get_gse_map_no_pos()
        
    def get_gse_map_no_pos(self) -> Dict[str, List[int]]:
        gse_map_no_pos = defaultdict(list)  # For finding lemmas when the POS tag doesn't match in gse_map
        for (lemma, _), difficulties in self.gse_map.items():
            gse_map_no_pos[lemma].extend(difficulties)
        return dict(gse_map_no_pos)
    
    def get_score(self, phrase: str, scoring_type='mean') -> Optional[float]:
        try:
            scores = self.gse_map_no_pos[phrase]
        except KeyError:
            lemma = get_lemma(phrase)
            try:  # Check with lemma tag if not found
                scores = self.gse_map_no_pos[lemma]
            except KeyError:
                return 'B2'
        score = num2cefr(gse_to_cefr9(get_gse_score(scores, scoring_type)))
        return score
    
    
    
    
    
    
    
    
    
    
    
    
    