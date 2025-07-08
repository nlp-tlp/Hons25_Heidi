import logging
import pickle
import re
from rapidfuzz import process, fuzz

from ..pkl.skb import SKB

FUZZY_SKB_PATH = "databases/other/fuzzy_types.pkl"

class Fuzzy_SKB:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.texts: dict[str, any] = {}

    def parse(self, skb: SKB, max_nodes: int = None, clear_previous: bool = True):
        if clear_previous:
            self.texts = {}

        for i, (node_id, node) in enumerate(skb.get_entities().items()):
            if max_nodes is not None and i >= max_nodes:
                break

            text_fields = node.get_textual()
            if not text_fields:
                continue

            text = " | ".join(self.normalise_string(v) for v in text_fields.values())
            self.texts.setdefault(text, set()).add(type(node).__name__)

        self.logger.info(f"New collection size: {len(self.texts)}")

    def normalise_string(self, text: str):
        if not text:
            return ""

        text = text.lower()
        text = text.rstrip(".,") # Lots of entries that end with comma or dot point
        text = re.sub(r'\s+', ' ', text) # Some entries have double whitespaces
        return text

    def query(self, text, top_k=5, threshold=70, return_scores: bool = True):
        self.logger.info(f"Fuzzy and partial matching for '{text}'")
        matches = process.extract(
            text.lower(), self.texts.keys(), scorer=self.combined_score, limit=top_k
        )

        return [
            (m[0], m[1], list(self.texts[m[0]])) if return_scores else (m[0], list(self.texts[m[0]]))
            for m in matches if m[1] >= threshold
        ]

    @staticmethod
    def combined_score(s1, s2, *, processor=None, score_cutoff=None):
        full = fuzz.ratio(s1, s2)
        partial = fuzz.partial_ratio(s1, s2)

        query_words = set(s1.lower().split())
        candidate_words = set(s2.lower().split())
        extra_words = len(candidate_words - query_words)

        score = (0.05 * full + 0.95 * partial) * (0.95 ** extra_words)

        if score_cutoff is not None and score < score_cutoff:
            return None
        return score

    def save_pickle(self, filepath: str = FUZZY_SKB_PATH):
        with open(filepath, 'wb') as f:
            pickle.dump(self.texts, f)

    def load_pickle(self, path: str = FUZZY_SKB_PATH):
        with open(path, "rb") as f:
            self.texts = pickle.load(f)
