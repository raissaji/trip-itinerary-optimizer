import streamlit as st
import pandas as pd
from cosine_retrieval import get_similar_recs
from ranking import rank_by_expense
import df_operations
import llm_generation

# --------------------
# Load data
# --------------------
ACCOM_CSV_PATH = "../data/accommodations.csv"
ACTIVITIES_CSV_PATH = "../data/activities.csv"
FOOD_CSV_PATH = "../data/food.csv"

hotel_df = pd.read_csv(ACCOM_CSV_PATH)
activity_df = pd.read_csv(ACTIVITIES_CSV_PATH)
food_df = pd.read_csv(FOOD_CSV_PATH)

st.set_page_config(page_title="Trip Itinerary Optimizer", layout="wide")
st.title("🧳 Trip Itinerary Optimizer")

# --------------------
# Helper functions
# --------------------

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


def build_itinerary(accommodations, activities, food, n):
    itinerary = []
    for i in range(n):
        itinerary.append({
            "lodging": accommodations[0],
            "activity": activities[i % len(activities)],
            "food": food[i % len(food)],
        })
    return itinerary

# --------------------
# Sidebar inputs
# --------------------
st.sidebar.header("Trip Preferences")

n_days = st.sidebar.number_input("Number of days", min_value=1, max_value=14, value=2)
k = st.sidebar.number_input("Top-k candidates", min_value=1, max_value=5, value=2)

accom_pref = st.sidebar.text_area("Accommodation preferences")
accom_budget = st.sidebar.number_input("Accommodation budget (€/night)", value=100.0)

activity_pref = st.sidebar.text_area("Activity preferences")
activity_budget = st.sidebar.number_input("Activity budget (€/day)", value=50.0)

food_pref = st.sidebar.text_area("Food preferences")
food_budget = st.sidebar.number_input("Food budget (€/day)", value=50.0)

run_button = st.sidebar.button("Generate itinerary")

# --------------------
# Main logic
# --------------------
if run_button:
    st.header("🔍 Retrieval Results")

    top_k_accom = get_similar_recs(accom_pref, "accommodations", k, hotel_df)
    st.subheader("Top-k Accommodations (Cosine Similarity)")
    st.dataframe(top_k_accom)

    top_k_activities = get_similar_recs(activity_pref, "activities", k, activity_df)
    st.subheader("Top-k Activities")
    st.dataframe(top_k_activities)

    top_k_food = get_similar_recs(food_pref, "food", k, food_df)
    st.subheader("Top-k Food")
    st.dataframe(top_k_food)

    st.header("💰 Budget Ranking")

    ranked_accom = rank_by_expense(top_k_accom, accom_budget, "accomodations")
    st.subheader("Ranked Accommodations")
    st.dataframe(ranked_accom)

    ranked_activities = rank_by_expense(top_k_activities, activity_budget, "activities")
    st.subheader("Ranked Activities")
    st.dataframe(ranked_activities)

    clean_food = clean_food_price_column(top_k_food)
    ranked_food = rank_by_expense(clean_food, food_budget, "food")
    st.subheader("Ranked Food")
    st.dataframe(ranked_food)

    st.header("🗺️ Itinerary Construction")

    accom_texts = df_operations.df_to_text_promptbuilding(top_k_accom, "accommodations")
    activity_texts = df_operations.df_to_text_promptbuilding(top_k_activities, "activities")
    food_texts = df_operations.df_to_text_promptbuilding(top_k_food, "food")

    itinerary = build_itinerary(accom_texts, activity_texts, food_texts, n_days)
    st.subheader("Structured Itinerary (Python object)")
    st.json(itinerary)

    st.header("✍️ LLM Prompt")
    prompt = llm_generation.make_narration_prompt(itinerary, n_days)
    st.code(prompt)

    st.header("🧠 LLM Output")
    response = llm_generation.generate_response(prompt)
    st.write(response)

else:
    st.info("Fill in preferences on the left and click **Generate itinerary**.")
