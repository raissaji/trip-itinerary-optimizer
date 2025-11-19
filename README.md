# Trip Itinerary Optimizer

**CS 4701: Practicum in Artificial Intelligence**

A smart trip planner that combines **information retrieval**, **machine learning**, and **optimization** to build travel itineraries that match a user’s preferences (budget, activities, dates, etc.) while trying to minimize cost and maximize “fun.”

---

## Project Overview

Planning a trip is annoying: you have to search for activities, lodging, restaurants, transportation, and then somehow fit everything into a schedule and budget.

Our goal is to build a system that:

1. Takes a **user profile + preferences** (destination, dates, budget, interests, travel style, etc.).
2. Finds **relevant trip options** (activities, attractions, lodging, flights, etc.) from a data source.
3. Uses **AI + optimization** to assemble one or more **candidate itineraries** that best match the user’s needs.

The project is split into two main parts:

- **Part 1 – Information Retrieval System**
- **Part 2 – Itinerary Optimization**

---

## Part 1: Information Retrieval

Given a user query with `n` features (destination, dates, budget range, activity type, etc.), we:

1. Represent the query and candidate trip items as vectors.
2. Compute similarity between the query and items.
3. Return the most relevant options to feed into the optimization stage.

Planned techniques:

- Text preprocessing and vectorization  
- Similarity measures:
  - Jaccard similarity
  - Cosine similarity
- Machine learning methods:
  - SVD-based embeddings
  - k-Nearest Neighbors search / classifiers

Potential data sources (subject to terms of service and feasibility):

- Public travel datasets
- Online travel/review platforms (e.g., TripAdvisor-style data)
- Synthetic data we generate if needed

---

## Part 2: Itinerary Optimization

Once we have a set of candidate trip options, we want to construct **day-by-day itineraries** that:

- Respect the **user’s constraints** (budget, dates, opening hours, travel time).
- Attempt to **maximize experience value** (number/quality of sights/activities).
- Try to **minimize total cost**.

Ideas we are exploring:

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
├── notebooks/              # Experiments and prototyping
├── src/
│   ├── retrieval/          # IR & similarity code (Part 1)
│   ├── optimization/       # Itinerary optimization (Part 2)
│   ├── models/             # ML models, embeddings, etc.
│   ├── interface/          # CLI / web frontend
│   └── utils/              # Shared helpers
├── tests/                  # Unit tests
├── README.md               # (this file)
└── requirements.txt        # Python dependencies
