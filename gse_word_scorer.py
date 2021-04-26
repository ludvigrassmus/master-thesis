import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.lemmatizer import Lemmatizer

from src.en import spacy_nlp_model


@dataclass
class GseScore:
    indices: List[int]
    value: Optional[float]
    pos: str


INVALID_LEMMAS = {'be', 'have', 'not'}
INVALID_SPACY_TAGS = {"BES", "HVS", "PRP$", "PDT", "WDT", "WP", "WP$", "WRB"}
VALID_POS_TAGS = {'VERB', 'ADV', 'ADJ', 'NOUN'}

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


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
    """
    GSE     22   30  36  43  51  59  67  76  85  90
    CEFR-9  -.5  0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5
    step size  8    6   7   8   8   8   9   9   5
    sections     |   |   |           |        |

    We don't have a lower bound for <A1, so we assume the same slope as A1

    Since the step size of CEFR-9 is always 1:

    formula: (score_gse - section_start_gse) / step_size_gse + section_start_cefr9

    = score_gse / step_size_gse + (section_start_cefr9 - section_start_gse / step_size_gse)

    <= 22: -.5 - 22/8 = -3.25
    <= 36: 0.5 - 30/6 = -4.5
    <= 43: 1.5 - 36/7 = -51/14
    <= 67: 2.5 - 43/8 = -2.875
    <= 85: 5.5 - 67/9 = -35/18
    > 85:  7.5 - 85/5 = -9.5
    """
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


class GseWordScorer:

    def __init__(self, gse_score_path: str):
        self.gse_map = get_gse_map(gse_score_path)
        self.gse_map_no_pos = self.get_gse_map_no_pos()

    def get_gse_map_no_pos(self) -> Dict[str, List[int]]:
        gse_map_no_pos = defaultdict(list)  # For finding lemmas when the POS tag doesn't match in gse_map
        for (lemma, _), difficulties in self.gse_map.items():
            gse_map_no_pos[lemma].extend(difficulties)
        return dict(gse_map_no_pos)

    def __call__(self, text, scoring_type='mean') -> List[GseScore]:
        word_difficulties = []
        nlp = spacy_nlp_model(text)
        for token in nlp:
            lemma = token.lemma_.lower()
            if lemma in INVALID_LEMMAS:
                continue

            tag = token.tag_
            if tag in INVALID_SPACY_TAGS:
                continue

            pos = token.pos_
            if pos not in VALID_POS_TAGS:
                continue

            start_index = token.idx
            end_index = start_index + len(token)
            score = self.get_score(str(token), pos, scoring_type)
            word_difficulties.append(GseScore(indices=[start_index, end_index], value=score, pos=pos))
        return word_difficulties

    def get_score(self, phrase: str, pos: str, scoring_type) -> Optional[float]:
        try:
            scores = self.gse_map[(phrase, pos)]
        except KeyError:
            lemma = self.get_lemma(phrase, pos)
            try:  # Check with POS tag
                scores = self.gse_map[(lemma, pos)]
            except KeyError:
                try:
                    scores = self.gse_map_no_pos[phrase]
                except KeyError:
                    try:  # Check without POS tag if not found
                        scores = self.gse_map_no_pos[lemma]
                    except KeyError:
                        return None
        score = gse_to_cefr9(get_gse_score(scores, scoring_type))
        return score

    @staticmethod
    def get_lemma(phrase, pos) -> str:
        lemmas = lemmatizer(phrase, pos)
        if len(lemmas) > 1:
            logging.warning(f"Found multiple lemmas for {phrase}, {pos}: {lemmas}")
            # If this ever happens, we should probably think of a better way to do this
        return lemmas[0]

    def score_list(self, phrases: List[str], pos: str, scoring_type: str) -> List[float]:
        result = []
        for phrase in phrases:
            result.append(self.get_score(phrase, pos, scoring_type))
        return result
