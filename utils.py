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
pinecone.init(api_key=pinecone_api, environment=environment)

EMBEDDING_MODEL = "text-embedding-ada-002"
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

hardcode= {"curiosity": "Never stop looking for a better way of doing things.", 
           "nimbleness": "Building businesses is fast-paced and fluid",
           "generousity": "Sharing knowledge and networks builds better businesses",
           "Seeking diversity": "The best ideas aren’t always your own",
           "Embracing Data": "Decisions should be driven by analytics and customer insights."}
df = pd.read_csv('parameter.csv')
df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
df.to_csv('embeddings.csv')
df.info(show_counts=True)   
ind = [i for i in range(0, len(df))]
df['embedding_id'] = ind
df.head()

embeddings = df['embedding']
embedding_ids = df['embedding_id'].astype("string")
texts = df['text']

index = pinecone.Index(index_name='embedded')

index.delete(deleteAll='true')
# Create a list of dictionaries with the data
index.upsert(list(zip(embedding_ids, embeddings)))

content_mapped = dict(zip(embedding_ids.astype(int),texts))

def query_article(query, top_k=1):
    '''Queries an article using its title in the specified
     namespace and prints results.'''

    # Create vector embeddings based on the title column
    embedded_query = openai.Embedding.create(
                                            input=query,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']

    # Query namespace passed as parameter using title vector
    query_result = index.query(embedded_query,
                                      top_k=top_k)
    print(f'\nMost similar results to {query}:')
    if not query_result.matches:
        print('no query result')
    # print(query_result.matches)
    matches = query_result.matches
    ids = [int(res.id) for res in matches]
    scores = [res.score for res in matches]
    res = []
    for i in ids:
      res.append(content_mapped[int(i)])
    return res
with st.form(key='form'):
    prompt = st.selectbox("Select Parameters",["Growth", "Attitude to feedback", "Humility", "Attitude to failure", "Perception of Potential", "Curiosity", "Attitude to complexity", "Agility", "Collobaration"])
    num = st.number_input("Enter Score:", min_value=0, max_value=100)
    value = st.selectbox("Select values", 
                                (['curiosity', 'nimbleness', 'generousity', 'Seeking diversity', 'Embracing Data']))

    submit_button = st.form_submit_button(label='Submit')

query_output = query_article(prompt)
print(query_output)

def apply_prompt_template(question: str, score: num, prompt: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        # You are an interviewer, \
        # Your responsibility is to ask 5 thorough questions on the parameter above and the score, \
        # Where the candidate has scored {score} marks out of 100. \
        # The questions should not repeat, question the candidate's value of {question}, \
        # keeping in mind his marks in the above-mentioned parameter. \
        # The meaning of {question} is {hardcode[question]}. 
        You are an interviewer you are given a task to ask 5 questions, \
        to evaluate {question} of candidate who is {score} percent {prompt} .
        """
    return prompt

messages = list(
map(lambda chunk: {
    "role": "user",
    "content": chunk
}, query_output))

#params = ["Growth", "Attitude to feedback", "Humility", "Attitude to failure", "Perception of Potential", "Curiosity", "Attitude to complexity", "Agility", "Collobaration"]

question = apply_prompt_template(value, num, prompt)
print(question)
messages.append({"role": "user", "content": question})

if submit_button:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        n=5,
        stop=None,
        temperature=0.7,  # High temperature leads to a more creative response.
    )

    prediction = response['choices'][0]['message']['content']

    st.write(prediction)

print(question)