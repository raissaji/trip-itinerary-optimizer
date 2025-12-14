# from typing import List, Set, Tuple
# import pandas as pd
# import re
# from cosine_retrieval import hotel_text
# import df_operations as df_operations

# WORD_RE = re.compile(r"[a-zA-Z]+")

# def tokenize(text: str) -> Set[str]:
#     """
#     Very simple tokenizer: lowercase and keep alphabetic tokens only.
#     """
    
#     return set(WORD_RE.findall(text.lower()))

# def build_corpus_tokens(df: pd.DataFrame) -> List[Set[str]]:
#     """
#     For each hotel, build a set of tokens representing it.
#     """

#     corpus_tokens: List[Set[str]] = []
#     for _, row in df.iterrows():
#         text = df_operations.row_to_text(row)
#         corpus_tokens.append(tokenize(text))
#     return corpus_tokens

# def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
#     """
#     Compute Jaccard similarity between two token sets.
#     """

#     if not a and not b:
#         return 0.0
#     inter = len(a & b)
#     union = len(a | b)
#     return inter / union if union > 0 else 0.0

# def retrieve_hotels(
#     user_query: str,
#     df: pd.DataFrame,
#     corpus_tokens: List[Set[str]],
#     top_k: int = 10
# ) -> Tuple[pd.DataFrame, List[float]]:
#     """
#     Given a user query and precomputed corpus token sets, return the top_k most similar hotels.
#     """

#     query_tokens = tokenize(user_query)
#     sims = [jaccard_similarity(query_tokens, doc_tokens) for doc_tokens in corpus_tokens]
#     indexed = list(enumerate(sims))
#     indexed.sort(key=lambda x: x[1], reverse=True)
#     top_idx = [i for i, s in indexed[:top_k]]
#     top_sims = [s for i, s in indexed[:top_k]]
#     top_df = df.iloc[top_idx].copy().reset_index(drop=True)
#     top_df["similarity"] = top_sims
#     return top_df, top_sims


# jaccard_retrieval.py

import pandas as pd
import re
from typing import Set


def tokenize(text: str) -> Set[str]:
    """
    Lowercase, remove punctuation, split into tokens.
    """
    if pd.isna(text):
        return set()

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    """
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def get_similar_recs(
    query: str,
    domain: str,
    k: int,
    df: pd.DataFrame,
    text_col: str = "description"
) -> pd.DataFrame:
    """
    Retrieve top-k rows using Jaccard similarity.

    Args:
        query: user preference text
        domain: 'accommodations' | 'activities' | 'food'
        k: number of results
        df: source dataframe
        text_col: column to compare text against

    Returns:
        Top-k dataframe sorted by similarity
    """

    query_tokens = tokenize(query)

    df = df.copy()
    df["similarity"] = df[text_col].apply(
        lambda x: jaccard_similarity(query_tokens, tokenize(x))
    )

    df = df.sort_values("similarity", ascending=False)

    return df.head(k).reset_index(drop=True)
