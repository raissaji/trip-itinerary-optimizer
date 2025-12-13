from retrieval import get_similar_recs
from ranking import rank_by_expense

def run_demo():
    user_accom_pref = input("Describe the kind of accommodation you prefer to live in. What is your maximum budget per night?")
    user_activity_pref = input("Describe the kinds of activites you enjoy doing. What is your maximum budget per day?")
    # user_food_pref = input("TBD")

    # Retrieval
    retrieved_data = get_similar_recs(user_accom_pref, "accommodations")
    print(retrieved_data)

    # Ranking / optimization
    # ranked_df = rank_by_expense(retrieved_df, budget)
    # print("Ranked recommendations (after applying budget-based ranking):")
    # for idx, row in ranked_df.head(5).iterrows():
    #     print(f"{idx+1}. {row['name']} in {row['city']} - "
    #           f"${row['price_per_night']}/night, rating {row['rating']}, "
    #           f"similarity={row['similarity']:.2f}")
    #     print(f"   (budget diff: {row['budget_diff']:.1f})")
    # return ranked_df

if __name__ == "__main__":
    run_demo()