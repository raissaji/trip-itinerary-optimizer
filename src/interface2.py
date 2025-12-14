import streamlit as st
import pandas as pd

from cosine_retrieval import get_similar_recs
import jaccard_retrieval
from ranking import rank_by_expense
import df_operations
import llm_generation

ACCOM_CSV_PATH = "../data/accommodations.csv"
ACTIVITIES_CSV_PATH = "../data/activities.csv"
FOOD_CSV_PATH = "../data/food.csv"

hotel_df = pd.read_csv(ACCOM_CSV_PATH)
activity_df = pd.read_csv(ACTIVITIES_CSV_PATH)
food_df = pd.read_csv(FOOD_CSV_PATH)


def build_itinerary(accomodations, activities, food, n):
    itinerary = []
    for i in range(n):
        itinerary.append({
            "lodging": accomodations[0],
            "activity": activities[i],
            "food": food[i]
        })
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


st.set_page_config(page_title="Trip Itinerary Optimizer", layout="wide")

st.title("🧭 Trip Itinerary Optimizer")
st.markdown("Explore retrieval, ranking, and LLM generation step by step.")

# ===================== Sidebar Inputs =====================
st.sidebar.header("User Preferences")

n_days = st.sidebar.number_input("Number of days", min_value=1, max_value=14, value=2)
k = st.sidebar.number_input("Top‑K results", min_value=1, max_value=10, value=2)

retrieval_method = st.sidebar.radio(
    "Retrieval method",
    options=["Cosine Similarity", "Jaccard Similarity"]
)

accomodation_pref = st.sidebar.text_area("Accommodation preference")
accomodation_budget = st.sidebar.number_input("Accommodation budget (€ / night)", min_value=0.0)

activity_pref = st.sidebar.text_area("Activity preference")
activity_budget = st.sidebar.number_input("Activity budget (€ / day)", min_value=0.0)

food_pref = st.sidebar.text_area("Food preference")
food_budget = st.sidebar.number_input("Food budget (€ / day)", min_value=0.0)

run_button = st.sidebar.button("🚀 Generate Itinerary")

# ===================== Retrieval Switch =====================
def retrieve(pref, domain, k, df):
    if retrieval_method == "Cosine Similarity":
        return get_similar_recs(pref, domain, k, df)
    else:
        return jaccard_retrieval.get_similar_recs(pref, domain, k, df)


# ===================== Main Pipeline =====================
if run_button:
    st.header("🔎 Retrieval Results")

    top_k_accom = retrieve(accomodation_pref, "accommodations", k, hotel_df)
    top_k_acts = retrieve(activity_pref, "activities", k, activity_df)
    top_k_food = retrieve(food_pref, "food", k, food_df)

    st.subheader("Top‑K Accommodations")
    st.dataframe(top_k_accom)

    st.subheader("Top‑K Activities")
    st.dataframe(top_k_acts)

    st.subheader("Top‑K Food")
    st.dataframe(top_k_food)

    st.header("💰 Budget Ranking")

    ranked_accom = rank_by_expense(top_k_accom, accomodation_budget, "accomodations")
    ranked_acts = rank_by_expense(top_k_acts, activity_budget, "activities")

    clean_food_df = clean_food_price_column(top_k_food)
    ranked_food = rank_by_expense(clean_food_df, food_budget, "food")

    st.subheader("Ranked Accommodations")
    st.dataframe(ranked_accom)

    st.subheader("Ranked Activities")
    st.dataframe(ranked_acts)

    st.subheader("Ranked Food")
    st.dataframe(ranked_food)

    st.header("🗺️ Itinerary Construction")

    accom_texts = df_operations.df_to_text_promptbuilding(top_k_accom, "accommodations")
    act_texts = df_operations.df_to_text_promptbuilding(top_k_acts, "activities")
    food_texts = df_operations.df_to_text_promptbuilding(top_k_food, "food")

    itinerary = build_itinerary(accom_texts, act_texts, food_texts, n_days)

    st.subheader("Structured Itinerary Object")
    st.json(itinerary)

    st.header("🧠 LLM Prompt & Response")

    prompt = llm_generation.make_narration_prompt(itinerary, k)
    st.subheader("Prompt")
    st.code(prompt)

    response = llm_generation.generate_response(prompt)

    st.subheader("LLM Response")
    st.success(response)
