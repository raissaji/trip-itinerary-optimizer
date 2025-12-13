from typing import List, Set, Tuple
import pandas as pd
import re
from retrieval import hotel_text

WORD_RE = re.compile(r"[a-zA-Z]+")

def tokenize(text: str) -> Set[str]:
    """
    Very simple tokenizer: lowercase and keep alphabetic tokens only.
    """
    
    return set(WORD_RE.findall(text.lower()))

def build_corpus_tokens(df: pd.DataFrame) -> List[Set[str]]:
    """
    For each hotel, build a set of tokens representing it.
    """

    corpus_tokens: List[Set[str]] = []
    for _, row in df.iterrows():
        text = hotel_text(row)
        corpus_tokens.append(tokenize(text))
    return corpus_tokens

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two token sets.
    """

    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def retrieve_hotels(
    user_query: str,
    df: pd.DataFrame,
    corpus_tokens: List[Set[str]],
    top_k: int = 10
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Given a user query and precomputed corpus token sets, return the top_k most similar hotels.
    """

    query_tokens = tokenize(user_query)
    sims = [jaccard_similarity(query_tokens, doc_tokens) for doc_tokens in corpus_tokens]
    indexed = list(enumerate(sims))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_idx = [i for i, s in indexed[:top_k]]
    top_sims = [s for i, s in indexed[:top_k]]
    top_df = df.iloc[top_idx].copy().reset_index(drop=True)
    top_df["similarity"] = top_sims
    return top_df, top_sims
