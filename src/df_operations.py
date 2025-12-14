"""Returns list of strings, where each string is a row in the df """
def df_to_text(df, feature): 
    texts = [row_to_text(row, feature) for _, row in df.iterrows()]
    return texts


def df_to_text_promptbuilding(df, feature):
    """
    Convert dataframe rows into short, TinyLlama-friendly text for LLM.
    Each entry will be 1–2 sentences, including name/location and cost.
    """
    texts = []
    for _, row in df.iterrows():
        if feature == "accommodations":
            text = f"{row['name']}, {row['location']}. {row['type'].capitalize()}. Cost: {row['euros_per_night']}€/night."
        elif feature == "activities":
            text = f"{row['location']}. {row['description'].split('.')[0]}. Cost: {row['euros_per_person']}€/person."
        elif feature == "food":
            text = f"{row['location']}. {row['type'].capitalize()}. {row['cuisine']}. Cost: {row['euros_per_person']}€/person."
        texts.append(text)
    return texts

def row_to_text(row, feature):
    if feature == "accommodations":
        return (
            f"Name: {row['name']}."
            f"Location: {row['location']}."
            f"Type: {row['type']}."
            f"Bathroom Type: {row['bathroom_type']}."
            f"Description: {row['description']}."
            f"Amenities: {row['amenities']}."
            f"Cost: {row['euros_per_night']} per night."
        )
    elif feature == "activities":
        return (
            f"Location: {row['location']}."
            f"Description: {row['description']}."
            f"Cost: {row['euros_per_person']} per person."
        )
    elif feature == "food":
        return (
            f"Location: {row['location']}."
            f"Type: {row['type']}."
            f"Cuisine: {row['cuisine']}."
            f"Description: {row['description']}."
            f"Cost: {row['euros_per_person']} per person." )