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

def dataprep(file, output):
    df = pd.read_csv(file)
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv(output)
    df.info(show_counts=True)
    ind = [i for i in range(0, len(df))]
    df['embedding_id'] = ind
    return df

df = dataprep('parameter.csv', 'embeddings.csv')
embeddings = df['embedding']
embedding_ids = df['embedding_id'].astype("string")
texts = df['text']

index = pinecone.Index(index_name='embedded')

index.delete(deleteAll='true')
# Create a list of dictionaries with the data
index.upsert(list(zip(embedding_ids, embeddings)), namespace='parameters')

content_mapped = dict(zip(embedding_ids.astype(int),texts))

def query_article(query, name, top_k=1):
    '''Queries an article using its title in the specified
     namespace and prints results.'''

    # Create vector embeddings based on the title column
    embedded_query = openai.Embedding.create(
                                            input=query,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']

    # Query namespace passed as parameter using title vector
    query_result = index.query(embedded_query,
                               namespace=name,
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
query_param = query_article(prompt, 'parameters')
#query_val = query_article(value, 'values')
print(query_param)
#print(query_val)

def apply_prompt_template(question: str, score: num) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above parameter, where the candidate scored {score} out of 100, ask 5 interview question based on it, that questions the candidate's value of {question}.Here the meaning of {question} is {hardcode[question]}
    """
    return prompt

messages = list(
map(lambda chunk: {
    "role": "user",
    "content": chunk
}, query_param))

question = apply_prompt_template(value, num)
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
