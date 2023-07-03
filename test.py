from getpass import getpass
from openai.embeddings_utils import get_embedding
import pandas as pd
import openai
import pinecone
import warnings
import os
import streamlit as st

openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone_api = os.environ["PINECONE_API_KEY"]
environment = os.environ["PINECONE_ENV"]


df = pd.read_csv('Book.csv')
num = 1
category = input('Enter Category: ')
prompt = f'''
Humility: The ability to recognize and accept one's limitations and mistakes without arrogance or excessive pride.
Growth: The continuous process of personal or professional development and improvement over time.
Resilience: The capacity to bounce back and recover quickly from setbacks, adversity, or difficult situations.
Attitude to Failure: The mindset and perspective one holds towards failures, seeing them as learning opportunities rather than permanent setbacks.
Fixed Beliefs: Strongly held opinions or convictions that are resistant to change, even in the face of contradictory evidence.
Curiosity: A strong desire to explore, learn, and seek new knowledge or experiences.
Blame: Assigning responsibility or fault to someone for a particular situation or outcome.
Agility: The ability to adapt quickly and effectively to changing circumstances or challenges.
Collaboration: Working together with others towards a common goal, sharing ideas, resources, and responsibilities.

In a growth mindset Assessment Test, the candidate {df.Firstname[num]}  has performed as follows:

Individual Category:
	
Humility: {df.H_hml_rating[num]}
Growth: {df.G_hml_rating[num]}
Resilience: {df.R_hml_rating[num]}
Attitude to Failure: {df.ATF_hml_rating[num]}
Fixed Beliefs: {df.FB_hml_rating[num]}
Curiosity: {df.C_hml_rating[num]}
Blame: {df.B_hml_rating[num]}
Agility: {df.AG_hml_rating[num]}
Collaboration: {df.COL_hml_rating[num]}
	
When compared against his team, his relative scores are:

Humility: {df.H_team_relative_pos[num]}
Growth: {df.G_team_relative_pos[num]}
Resilience: {df.R_team_relative_pos[num]}
Attitude to Failure: {df.ATF_team_relative_pos[num]}
Fixed Beliefs: {df.FB_team_relative_pos[num]}
Curiosity: {df.C_team_relative_pos[num]}
Blame: {df.B_team_relative_pos[num]}
Agility: {df.AG_team_relative_pos[num]}
Collaboration: {df.COL_team_relative_pos[num]}

And his team overall scores are:

Humility: {df.H_team_overall_pos[num]}
Growth: {df.G_team_relative_pos[num]}
Resilience: {df.R_team_relative_pos[num]}
Attitude to Failure: {df.ATF_team_relative_pos[num]}
Fixed Beliefs: {df.FB_team_relative_pos[num]}
Curiosity: {df.C_team_relative_pos[num]}
Blame: {df.B_team_relative_pos[num]}
Agility: {df.AG_team_relative_pos[num]}
Collaboration: {df.COL_team_relative_pos[num]}

Comment and give me an interview question to ask the candidate, based of the data on their individual, team-relative and team-overall grades of {category}
'''

# messages = list(
# map(lambda chunk: {
#     "role": "system",
#     "content": chunk
# }, system))


# def apply_prompt_template(category: str) -> str:
#     """
#         A helper function that applies additional template on user's question.
#         Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
#     """
#     prompt = f"""
#         Comment and give me an interview question to ask the candidate, based of the data on his individual, team-relative and team-overall grades of {category}.
#     """
#     return prompt

# question = apply_prompt_template(category)
# messages.append({"role": "user", "content": question})

# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=messages,
#     max_tokens=500,
#     n=5,
#     stop=None,
#     temperature=0.7,
# )

print(system)

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=system,
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response['choices'][0]['text'])


