import pandas as pd
import numpy as np

def rank_by_expense(df_top: pd.DataFrame, budget: float, preference: str) -> pd.DataFrame:
    """
    Rank the retrieved hotels by closeness to budget, then by price, then by rating (descending).
    """
    if preference == "accomodations": 
        df = df_top.copy()
        df["euros_per_night"] = pd.to_numeric(df["euros_per_night"], errors="coerce")
        df["budget_diff"] = df["euros_per_night"] - budget
        df = df.sort_values(
            ["budget_diff", "euros_per_night"],
            ascending=[True, True]
        )
    else: 
        df = df_top.copy()
        df["euros_per_person"] = pd.to_numeric(df["euros_per_person"], errors="coerce")
        df["budget_diff"] = df["euros_per_person"] - budget
        df = df.sort_values(
            ["budget_diff", "euros_per_person"],
            ascending=[True, True]
        )
    return df.reset_index(drop=True)
