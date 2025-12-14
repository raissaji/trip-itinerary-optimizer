# Southern Italy Trip Itinerary Optimizer

**CS 4701: Practicum in Artificial Intelligence**

A smart trip planner that combines **information retrieval**, **natural language processing**, and **optimization** to build travel itineraries for Southern Italy that match a user’s preferences (accommodation, activities, food) while staying within budget and maximizing fun (how well the itinerary matches the user's preferences).

---

## Project Overview

Planning a trip is annoying: you have to search for activities, lodging, restaurants, transportation, and then somehow fit everything into a schedule and budget.

Our goal is to build a system that:

1. Takes a **user profile + preferences** (destination, budget, interests, travel style, etc.).
2. Finds **relevant trip options** (activities, attractions, lodging, food, etc.) from various datasets.
3. Uses **AI/ML + optimization** to assemble one or more **candidate itineraries** that best match the user’s needs.

The project is split into two main parts:

- **Part 1 – Information Retrieval System**
- **Part 2 – Itinerary Optimization**

---

## Part 1: Information Retrieval

Given a user query with `4` features (accommodation style, accomodation budget, activities, activity budget, food preferences, food budget), we:

1. Embed the query and datasets as vectors. (note: we don't use embeddings for jaccard similarity) 
3. Compute similarity between the query and datasets. (compare embeddings for cosine similarity, compare query tokens and document tokens for jaccard similarity) 
4. Return the most relevant options, above an arbitrary threshold, to feed into the optimization stage.

Planned AI/ML tools and frameworks:

-  Use Sentence-BERT (SBERT) to encode query and documents into embeddings. 
- Similarity measures:
  - Jaccard similarity
  - Cosine similarity

Data sources:

- Generated through OpenAI Deep Research

---

## Part 2: Itinerary Optimization

Once we have a set of candidate trip options, we want to construct **day-by-day itineraries** that:

- Respect the **user’s constraints** (budget and other preferences).
- Attempt to **maximize experience value** (number/quality of sights/activities).
- Stay within **specified budget**.

Our technique: rank reccomendations by difference between item's budget and user's maximum budget. Smaller differences have higher ranking. 

---

## 🛠️ Tech Stack (Planned)

This may evolve as we implement, but our current plan:

- **Language:** Python
- **Core libraries (candidate list):**
  - `numpy`, `pandas` – data handling
  - `scikit-learn` – vectorization, SVD, KNN, etc.
  - `scipy` / optimization libraries – possible LP/heuristic implementations
- **Frontend:** simple web or notebook-based interface to enter preferences and view itineraries
  - streamlit

---

## 📁 Project Structure

```text
trip-itinerary-optimizer/
├── data/                   # Raw / processed travel data      
│   ├── accomodations.csv
│   ├── activites.csv
│   ├── food.csv
├── src/
│   ├── df_operations.py       # operations for processing dataframes(Part 1)
│   ├── jaccard_retrieval.py   # KNN search & compute jaccard similarity (Part 1)
│   ├── cosine_retrieval.py    # KNN search & compute cosine similarity  (Part 1)
│   ├── ranking.py             # optimize itinerary using budget (Part 2)
│   ├── llm_generation.py      #construct LLM prompt and generate response (part 2) 
│   ├── main.py                # Prompt for user query, facilitate overall process of info retrieval, budget optimization, building itinerary, generating and feeding LLM prompt
│   ├── interface2.py          # actual web frontend 
│   ├── interface1.py          # old frontend
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## Prerequisites
• Python 3.11 is required
## Installing
1) Clone the repo on your machine
```
git clone https://github.com/raissaji/trip-itinerary-optimizer.git
```
2) Install requirements
```
pip install -r requirements.txt
```
3) We use the TinyLlama-1.1B-Chat-v0.3 model, a surprisingly capable small model with 1.1B parameters (takes 0.7-0.8 GB of memory) and could run on CPU. Install the tinyllama model on your local directory with the following command: 
```
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
```
To run our program, run the following commands from the src directory:
```
pip install streamlit
streamlit run web_app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8502/).
```



