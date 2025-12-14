import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import df_operations

ACCOM_CSV_PATH = "../data/accommodations.csv"
ACTIVITIES_CSV_PATH = "../data/activities.csv"
FOOD_CSV_PATH = "../data/food.csv"

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_similar_recs(user_pref, feature,k, df):
    user_pref_embedding = model.encode(user_pref, convert_to_tensor=True)

    texts = df_operations.df_to_text(df, feature)
    item_embedding = model.encode(texts, convert_to_tensor=True)

    # Compute cosine similarity between user query and all items
    scores = util.cos_sim(user_pref_embedding, item_embedding)[0]

    # Get indices of top-k most similar items
    top_k = k #round(0.5 * len(df))
    top_scores, top_indices = torch.topk(scores, k=top_k)

    top_indices = top_indices.cpu().numpy()
    top_scores = top_scores.cpu().numpy()

    result = df.iloc[top_indices].copy()
    result["similarity"] = top_scores

    result.sort_values("similarity", ascending=False)


    return result.sort_values("similarity", ascending=False)

# def get_similar_recs_jaccard(user_pref, feature,k, df):
#      texts = df_operations.df_to_text(df, feature)
#      jaccard_list = []
#      for doc_text in texts: 
          