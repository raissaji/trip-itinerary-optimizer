import pandas as pd

def rank_by_expense(df_top: pd.DataFrame, budget: float) -> pd.DataFrame:
    """
    Rank the retrieved hotels by closeness to budget, then by price, then by rating (descending).
    """
    df = df_top.copy()
    df["budget_diff"] = (df["price_per_night"] - budget).abs()
    df = df.sort_values(
        ["budget_diff", "price_per_night", "rating"],
        ascending=[True, True, False]
    )
    return df.reset_index(drop=True)
