import pandas as pd

def load_hotels(csv_path: str) -> pd.Dataframe:
    """
    Load a hotels dataset from a CSV file.
    """

    df = pd.read_csv(csv_path)
    return df

def hotel_text(row) -> str:
     """
    Build a textual representation of a hotel that we will use for similarity search.
    """
     
    return (
        f"{row['name']} in {row['city']}. "
        f"{row['description']} Amenities: {row['amenities']}."
    )
    

