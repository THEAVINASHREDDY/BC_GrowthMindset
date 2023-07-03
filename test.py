from getpass import getpass
from openai.embeddings_utils import get_embedding
import pandas as pd
import openai
# import pinecone
# import warnings
import os
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
openai.api_key = os.environ["OPENAI_API_KEY"]


df = pd.read_csv('Book.csv')
with st.form(key='form'):
    name = st.selectbox("Select Candidate: ",df['Firstname'].values)
    category =st.selectbox("Choose Value: ", ['Humility', 'Growth', 'Resilience', 'Attitude to Failure', 'Fixed Beliefs', 'Curiosity', 'Blame', 'Agility', 'Collaboration'])
    submit_button = st.form_submit_button(label='Submit') 
num = df[df['Firstname'] == name].index.values

score_mapping = {'low': 1, 'medium': 2, 'high': 3}
individual_scores = [
    score_mapping[df.H_hml_rating[num].values[0].lower()],  # Convert to lowercase and update mapping usage
    score_mapping[df.G_hml_rating[num].values[0].lower()],
    score_mapping[df.R_hml_rating[num].values[0].lower()],
    score_mapping[df.ATF_hml_rating[num].values[0].lower()],
    score_mapping[df.FB_hml_rating[num].values[0].lower()],
    score_mapping[df.C_hml_rating[num].values[0].lower()],
    score_mapping[df.B_hml_rating[num].values[0].lower()],
    score_mapping[df.AG_hml_rating[num].values[0].lower()],
    score_mapping[df.COL_hml_rating[num].values[0].lower()]
]
# plt.figure(figsize=(10, 6))
categories = ['Humility', 'Growth', 'Resilience', 'Attitude to Failure', 'Fixed Beliefs', 'Curiosity', 'Blame', 'Agility', 'Collaboration']
fig = px.line_polar(df, r=individual_scores, theta=categories, line_close=True)
#fig.show()
# plt.bar(categories, individual_scores)
# plt.xlabel('Categories')
# plt.ylabel('Scores')
# plt.title('Candidate Scores in Individual Category')
# plt.xticks(rotation=45)



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

In a growth mindset Assessment Test where the candidate is scored based on the above parameters, the candidate {df.Firstname[num].values}  has performed as follows:

Individual Category:
	
Humility: {df.H_hml_rating[num].values}
Growth: {df.G_hml_rating[num].values}
Resilience: {df.R_hml_rating[num].values}
Attitude to Failure: {df.ATF_hml_rating[num].values}
Fixed Beliefs: {df.FB_hml_rating[num].values}
Curiosity: {df.C_hml_rating[num].values}
Blame: {df.B_hml_rating[num].values}
Agility: {df.AG_hml_rating[num].values}
Collaboration: {df.COL_hml_rating[num].values}
	
When compared against their team members, the candidates team-relative scores are:

Humility: {df.H_team_relative_pos[num].values}
Growth: {df.G_team_relative_pos[num].values}
Resilience: {df.R_team_relative_pos[num].values}
Attitude to Failure: {df.ATF_team_relative_pos[num].values}
Fixed Beliefs: {df.FB_team_relative_pos[num].values}
Curiosity: {df.C_team_relative_pos[num].values}
Blame: {df.B_team_relative_pos[num].values}
Agility: {df.AG_team_relative_pos[num].values}
Collaboration: {df.COL_team_relative_pos[num].values}

And their team-overall scores are:

Humility: {df.H_team_overall_pos[num].values}
Growth: {df.G_team_relative_pos[num].values}
Resilience: {df.R_team_relative_pos[num].values}
Attitude to Failure: {df.ATF_team_relative_pos[num].values}
Fixed Beliefs: {df.FB_team_relative_pos[num].values}
Curiosity: {df.C_team_relative_pos[num].values}
Blame: {df.B_team_relative_pos[num].values}
Agility: {df.AG_team_relative_pos[num].values}
Collaboration: {df.COL_team_relative_pos[num].values}

Comment and give me an interview question to ask the candidate, based of the data on their individual, team-relative and team-overall grades of {category}
The question should be with respect to the analysis you developed through your comment and their scores in the assessment.
Also explain your reasoning behind your question and its relevance to your comment.
Write comment, question and reasoning separately.
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

print(prompt)
print(num)
if submit_button:
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  st.plotly_chart(fig, use_container_width=True)
  st.write(response['choices'][0]['text'])
  


