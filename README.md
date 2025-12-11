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

Given a user query with `4` features (accommodation style, activities, food preferences, budget range), we:

1. Embed the query and datasets as vectors.
2. Compute similarity between the query and datasets.
3. Return the most relevant options, above an arbitrary threshold, to feed into the optimization stage.

Planned AI/ML tools and frameworks:

- Text preprocessing and vectorization using Sentence-BERT (SBERT)
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

Planned techniques (TBD)

- Heuristic or game-tree style search over possible itineraries  
- Monte Carlo tree search to simulate and evaluate many possible itineraries  
- Linear / integer programming formulations (e.g., select a subset of activities and assign them to time slots subject to constraints)

The output will be one or more **ranked itineraries** that the user can inspect and compare.

---

## 🛠️ Tech Stack (Planned)

This may evolve as we implement, but our current plan:

- **Language:** Python
- **Core libraries (candidate list):**
  - `numpy`, `pandas` – data handling
  - `scikit-learn` – vectorization, SVD, KNN, etc.
  - `scipy` / optimization libraries – possible LP/heuristic implementations
- **Frontend:** simple web or notebook-based interface to enter preferences and view itineraries

---

## 📁 Project Structure (Tentative)

```text
trip-itinerary-optimizer/
├── data/                   # Raw / processed travel data
├── src/
│   ├── retrieval/          # IR & similarity code (Part 1)
│   ├── optimizer/          # Itinerary optimization (Part 2)
│   ├── models/             # ML models, embeddings, etc.
│   ├── main/               # Prompt for user query and generating itinerary
│   ├── interface/          # CLI / web frontend
│   └── utils/              # Shared helpers
├── tests/                  # Unit tests
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## Prerequisites
• Python 3.11 is required
## Installing
Clone the repo on your machine
```
git clone https://github.com/raissaji/trip-itinerary-optimizer.git
```
Install requirements
```
pip install -r requirements.txt
```
