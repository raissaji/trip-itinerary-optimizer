import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

ACCOM_CSV_PATH = "../data/accommodations.csv"
ACTIVITIES_CSV_PATH = "../data/activities.csv"
FOOD_CSV_PATH = "../data/food.csv"

model = SentenceTransformer("all-MiniLM-L6-v2")

def row_to_text(row, feature):
    if feature == "accommodations":
        return (
            f"Name: {row['name']}."
            f"Location: {row['location']}."
            f"Coordinates: {row['coordinates']}."
            f"Type: {row['type']}."
            f"Bathroom Type: {row['bathroom_type']}."
            f"Description: {row['description']}."
            f"Amenities: {row['amenities']}."
            f"Euros per Night: {row['euros_per_night']}."
        )
    elif feature == "activities":
        return (
            f"Location: {row['location']}."
            f"Coordinates: {row['coordinates']}."
            f"Description: {row['description']}."
            f"Euros per Person: {row['euros_per_person']}."
        )
    else:
        pass

def get_similar_recs(user_pref, feature):
    user_pref_embedding = model.encode(user_pref, convert_to_tensor=True)

    if feature == "accommodations":
        df = pd.read_csv(ACCOM_CSV_PATH)
    elif feature == "activities":
        df = pd.read_csv(ACTIVITIES_CSV_PATH)
    else:
        df = pd.read_csv(FOOD_CSV_PATH)

    texts = [row_to_text(row, feature) for _, row in df.iterrows()]
    item_embedding = model.encode(texts, convert_to_tensor=True)

    # Compute cosine similarity between user query and all items
    scores = util.cos_sim(user_pref_embedding, item_embedding)[0]

    # Get indices of top-k most similar items
    top_k = round(0.6 * len(df))
    top_scores, top_indices = torch.topk(scores, k=top_k)

    top_indices = top_indices.cpu().numpy()
    top_scores = top_scores.cpu().numpy()

    result = df.iloc[top_indices].copy()
    result["similarity"] = top_scores

    return result.sort_values("similarity", ascending=False)