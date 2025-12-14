from cosine_retrieval import get_similar_recs
import llm_generation as llm_generation
from ranking import rank_by_expense
import pandas as pd
import jaccard_retrieval
import df_operations
import json

ACCOM_CSV_PATH = "../data/accommodations.csv"
ACTIVITIES_CSV_PATH = "../data/activities.csv"
FOOD_CSV_PATH = "../data/food.csv"

hotel_df = pd.read_csv(ACCOM_CSV_PATH)
acitvity_df = pd.read_csv(ACTIVITIES_CSV_PATH)
food_df = pd.read_csv(FOOD_CSV_PATH)

"""build itinerary (list of dictionaries: [{day1}, {day2}, ...] for n days 
    accomodations: list of texts for each accomodation option 
    activties: list of texts for each activities option
    food: list of texts for each food option"""
def build_itinerary(accomodations,activities,food, n):
    itinerary=[]
    for i in range(n):
        day_i_plan= {"lodging":accomodations[0], 
        "activity": activities[i], 
        "food":food[i] }
        itinerary.append(day_i_plan)
    return itinerary

def clean_food_price_column(df, col="euros_per_person"):
    df = df.copy()

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("€", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

    return df


def run_demo():
    k = 2

    #user preferences for accomodation, activity, food, and budget
    n_days = int(input("How many days do you want your itinerary to be? "))
    accomodation_pref = input("Describe the kind of accommodation you prefer to live in.")
    accomodation_budget = float(input("What is your maximum accomodation budget per night?"))

    activity_pref = input("Describe the kinds of activites you enjoy doing.")
    activity_budget = float(input("What is your maximum activity budget per day?"))

    food_pref = input("Describe what type of food you want to eat or restaurants you would like to eat at.")
    food_budget = float(input("What is your maximum food budget per day?"))

    

    #similarity metric: cosine similarity 
    # Retrieve top_k documents (as df) for accomodation, activites, and food
    pd.set_option("display.max_columns", None)

    top_k_accomodations = get_similar_recs(accomodation_pref, "accommodations",k, hotel_df) 
    print("TOP K ACCOMODATIONS -- COSINE SIM METRIC")
    print(top_k_accomodations)

    top_k_activities = get_similar_recs(activity_pref, "activities",k, acitvity_df) 
    print("TOP K ACTIVITIES")
    print(top_k_activities)
    # print(top_k_activities.columns)

    top_k_food = get_similar_recs(food_pref, "food",k, food_df) 
    print("TOP K FOOD")
    print(top_k_food)


    #budget optimization (rank top_k docs in terms of budget, price, rating) 
    ranked_accomodations = rank_by_expense(top_k_accomodations, accomodation_budget, "accomodations")
    print("RANKED ACCOMODATIONS")
    print(ranked_accomodations)

    ranked_activites = rank_by_expense(top_k_activities, activity_budget, "activities")
    print("RANKED ACTIVITIES")
    print(ranked_activites)
    print(top_k_activities["euros_per_person"].dtype)


    clean_food_df = clean_food_price_column(top_k_food)

    ranked_food = rank_by_expense(clean_food_df, food_budget, "food")
    print("RANKED FOOD")
    print(ranked_food)


    #convert dfs into lists of text, ideal for constructing llm prompt
    topk_accomodation_texts = df_operations.df_to_text_promptbuilding(top_k_accomodations, "accommodations") #convert retrieved data to list of doc text
    print(topk_accomodation_texts)
    
    topk_activity_texts = df_operations.df_to_text_promptbuilding(top_k_activities, "activities") #convert retrieved data to list of doc text
    print(topk_activity_texts)
    
    topk_food_texts = df_operations.df_to_text_promptbuilding(top_k_food, "food") #convert retrieved data to list of doc text
    print(topk_food_texts)
    
    #build itinerary (top-ranked rows of each category --> 1 day's itinerary)
    itinerary = build_itinerary(topk_accomodation_texts,topk_activity_texts,topk_food_texts, n_days)
    print("ITINERARY: ")
    print(itinerary)
    
    
    #build prompt for LLM
    print("BUILD LLM PROMPT")
    #select lodging, activites, and food for each day of itinerary
    prompt = llm_generation.make_narration_prompt(itinerary, k)
    print(prompt)

    # #put prompt into LLM model for natural language response
    print("SEND PROMPT TO LLM")
    response = llm_generation.generate_response(prompt)
    print("RESPONSE: " + response)
    print("END RESPONSE")

if __name__ == "__main__":
    run_demo()