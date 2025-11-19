from data_loader import load_hotels
from retrieval import build_corpus_tokens, retrieve_hotels
from ranking import rank_by_expense

CSV_PATH = "hotels.csv"

def run_demo():
    df = load_hotels(CSV_PATH)
    corpus_tokens = build_corpus_tokens(df)

    user_query = (
        "I want a quiet hotel in Barcelona near the beach "
        "with free breakfast and good wifi, budget around 180 per night "
        "and maybe some nightlife."
    )
    budget = 180

    print("User query:")
    print(user_query)
    print(f"Budget: ${budget} per night")
    print("-" * 60)

    # Retrieval
    retrieved_df, sims = retrieve_hotels(user_query, df, corpus_tokens, top_k=10)
    print("Top 10 hotels by similarity:")
    print(retrieved_df[["name", "city", "price_per_night", "rating", "similarity"]])
    print("-" * 60)

    # Ranking / optimization
    ranked_df = rank_by_expense(retrieved_df, budget)
    print("Ranked recommendations (after applying budget-based ranking):")
    for idx, row in ranked_df.head(5).iterrows():
        print(f"{idx+1}. {row['name']} in {row['city']} - "
              f"${row['price_per_night']}/night, rating {row['rating']}, "
              f"similarity={row['similarity']:.2f}")
        print(f"   (budget diff: {row['budget_diff']:.1f})")
    return ranked_df

if __name__ == "__main__":
    run_demo()
