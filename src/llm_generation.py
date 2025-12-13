import pandas as pd

#  #STEP 4: Construct prompt for LLM 
    #LLMs are sensitive to how information is presented.
    #A clear structure helps the model distinguish between the question and the reference material. 
    #Component4 combines the user's question with retrieved documents into a structured prompt. 
    #You can form a new prompt using the format of :
    #query + " Top documents:" + top1_doc_text + top2_doc_text + top3_doc_text.
"""Construct LLM prompt. Combine user query with retrieved top-k relevant documents into a structured prompt.
  Parameters: 
  query: user query (string)
  docs: df"""
def make_prompt(query:str, docs: pd.DataFrame)-> str: 
 # prompt = query + " Top documents: " + top1_doc_text + ", " + top2_doc_text + ", " + top3_doc_text
  # prompt = 
  # end_4 = time.perf_counter()
  # build_llm_prompt = end_4-start_4
  pass 


#STEP 5: Put prompt into LLM to generate a natural language response
    #model being used: TinyLlama-1.1B-Chat-v0.3, a surprisingly capable small model with 1.1B parameters 
    # (takes 0.7-0.8 GB of memory) and could run on CPU.
    #How to input prompt into model and output a response? 