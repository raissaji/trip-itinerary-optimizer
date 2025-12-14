import os
import pandas as pd
from typing import List, Set, Tuple
from llama_cpp import Llama

# from openai import OpenAI
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# response = client.responses.create(
#     model="gpt-5",
#     reasoning={"effort": "low"},
#     instructions="Talk like a pirate.",
#     input="Are semicolons optional in JavaScript?",
# )

# print(response.output_text)

#  #STEP 4: Construct prompt for LLM 
    #LLMs are sensitive to how information is presented.
    #A clear structure helps the model distinguish between the question and the reference material. 
    #Component4 combines the user's question with retrieved documents into a structured prompt. 
"""n is number of days itinerary should be"""
def make_narration_prompt(itinerary: list[dict], n: int) -> str:
    """
    itinerary: list of dicts, each dict has keys 'lodging', 'activity', 'food'
    n: number of days
    """
    itinerary_text = ""
    for i, day in enumerate(itinerary, start=1):
        itinerary_text += (
            f"Day {i}:\n"
            f"- Lodging: {day['lodging']}\n"
            f"- Activity: {day['activity']}\n"
            f"- Food: {day['food']}\n\n"
        )

#   prompt = f"""
# You are a travel writer.

# Selected itinerary:
# {itinerary_text.strip()}

# Write a friendly {n}-day itinerary based on the selections above.
# Use complete sentences and do NOT repeat the bullet points verbatim.
# Make it sound like a travel guide for a visitor.
# """

    prompt = f"""
      **INPUT DATA:**
Selected itinerary:
{itinerary_text.strip()}

**TASK:**
You are a friendly travel guide. Write a 2-day itinerary based on the INPUT DATA.

**CONSTRAINTS:**
1. Use a welcoming, narrative tone.
2. Use complete sentences.
3. DO NOT use bullet points or lists.
4. Output the result under the "ITINERARY START" line.

--- ITINERARY START ---"""
    return prompt.strip()

  

#STEP 5: Put prompt into LLM to generate a natural language response
#must download model to demo wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
llm = Llama(
    model_path= "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",    #Llama-3.2-3B-Instruct-Q4_K_M.gguf
    n_ctx=5000,  #1024 for bigger model
    n_threads=8,      # CPU threads (use for big model only)
    use_metal=True,    # GPU acceleration (use for big model only)
    chat_format="chatml", 
    temperature=0.5
)


"""Returns a natural language response to the user prompt through a LLM (uses llama API)"""
def generate_response(prompt):
  print("EXECUTING GENERATE RESPONSE METHOD")

  messages = [
    {"role": "system", "content": "You are a helpful travel writer."},
    {"role": "user", "content": prompt}
]
  result = llm.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.5
)
  
  string_result = str(result["choices"][0]["message"]["content"])
  return string_result

  






  # """Construct LLM prompt. Combine user query with retrieved top-k relevant documents into a structured prompt.
#   Parameters: 
#   query: user query (string)
#   docs: list of document text [doc1text, doc2,text,...,docktext]"""
# def make_prompt(query: str, docs: List[str]) -> str:
#     docs_block = "\n".join(
#         [f"{i+1}. {doc}" for i, doc in enumerate(docs)]
#     )

#     prompt = f"""
# You are a travel itinerary planner.

# User preferences:
# {query}

# Available options (use ONLY these options):
# {docs_block}

# Task:
# - Create a 3-day itinerary
# - You MUST select:
#   - ONE accommodation total
#   - ONE activity per day
#   - ONE food option per day
# - Use ONLY names from the options above
# - Do NOT leave any field blank

# EXAMPLE (format only, not real data):
# Day 1:
# - Lodging: Example Hotel
# - Activity: Example Museum
# - Food: Example Restaurant

# Output format:
# Day 1:
# - Loding:
# - Activity:
# - Food:

# Day 2:
# - Lodging:
# - Activity:
# - Food:
# """

#     return prompt.strip()



# def block(title, docs):
#   return "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(docs))
    


# def make_selection_prompt(
#     accom_docs: List[str],
#     activity_docs: List[str],
#     food_docs: List[str]
# ) -> str:
#     prompt = f"""
# You are selecting trip options.

# ACCOMMODATIONS:
# {block("accom", accom_docs)}

# ACTIVITIES:
# {block("act", activity_docs)}

# FOOD:
# {block("food", food_docs)}

# TASK:
# Return EXACTLY this JSON and NOTHING else.
# Use ONLY names that appear above.

# {{
#   "lodging": "",
#   "day1_activity": "",
#   "day1_food": "",
# }}
# """

#     return prompt.strip()
