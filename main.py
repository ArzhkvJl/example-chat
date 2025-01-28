import streamlit as st
import os
import sys
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.langchain import LangChainKnowledgeBase
from phi.run.response import RunResponse, RunEvent
import tweepy
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb


def create_vectorstore():
    urls = [
        "https://file.notion.so/f/f/c78cce49-cb8e-4e03-ba80-6c36906a72d6/d3688cc9-756f-4306-95ae-32e9fe94b233"
        "/The_End_Game_for_Oracles.pdf?table=block&id=18334261-8b12-809d-9df9-d55abf41f792&spaceId=c78cce49-cb8e-4e03"
        "-ba80-6c36906a72d6&expirationTimestamp=1738000800000&signature=8cGvb91itfxIz2u8NzbKYRPLHYPqFoKnxS1KWK-DA08"
        "&downloadName=The+End+Game+for+Oracles.pdf",
        "https://file.notion.so/f/f/c78cce49-cb8e-4e03-ba80-6c36906a72d6/47b518cb-d40d-4d2e-86b1-73d1d7d8d15e"
        "/Intro_To_Oracle_Validation_Services_.pdf?table=block&id=18334261-8b12-8012-ae74-c87d049176b7&spaceId=c78cce49"
        "-cb8e-4e03-ba80-6c36906a72d6&expirationTimestamp=1738000800000&signature"
        "=X7od3R9MqfXVlQmmP5ObW7GfeWTwH54k6GsGzFOYmLc&downloadName=Intro_To_Oracle_Validation_Services_.pdf",
        "https://file.notion.so/f/f/c78cce49-cb8e-4e03-ba80-6c36906a72d6/bae6c27f-5394-4b6c-93c5-4d8c66bd663e"
        "/How_To_build_an_Oracle_.pdf?table=block&id=18334261-8b12-806b-b281-c14d0b9d9f8f&spaceId=c78cce49-cb8e-4e03-ba80"
        "-6c36906a72d6&expirationTimestamp=1738000800000&signature=nCNl0b6eFnEU83g5uGEeeFghinDAarQvJaLqaU82duk"
        "&downloadName=How_To_build_an_Oracle_.pdf",
        "https://docs.eoracle.io/docs",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    hf_embeddings = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(hf_embeddings, breakpoint_threshold_type="percentile")
    doc_splits = text_splitter.split_documents(docs_list)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    return vectorstore


def create_prompt():
    # Load CSV containing topics and posts
    df = pd.read_csv(filepath_or_buffer="data/succinct_tweets.csv", on_bad_lines='skip', header=None)
    prompt = ("You are an AI assistant trained to generate Twitter posts based "
              "on the language patterns and structure found in an examples. "
              "Here are some examples:\n")
    for row in df.values:
        prompt += str(row)
    prompt += (f"Now, based on this pattern, "
               f"please create a Twitter reply for this post that is between 120-160 letters long.\n")
    return prompt


def generate_chat_responses(chat_completion):
    for chunk in chat_completion:
        if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
            if chunk.event == RunEvent.run_response:
                yield chunk.content


st.set_page_config(
    page_icon="💬",
    page_title="Chat App",
    layout="wide",
)

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")
twitter_api_key = st.sidebar.text_input("X (Twitter) Bearer Token", type="password")
os.environ['USER_AGENT'] = 'Agent-For-Twitter'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("Twitter reply Agent 🎈")

chromadb.api.client.SharedSystemClient.clear_system_cache()
retriever = create_vectorstore().as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)
agent = Agent(model=Groq(id="llama-3.3-70b-versatile"),
              knowledge_base=knowledge_base,
              description="You are an AI assistant trained to generate Twitter replies for posts",
              instructions=["Read a base post and generate a reply.",
                            "If post consists context about eOracle or eoracle, use knowledgebase and paste information about eOracle into reply.",
                            "Create a Twitter reply for this post that is between 120-160 letters long.",
                            "Use one or two smiles from your language patterns."])

if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.3-70b-versatile"

left, right = st.columns([2, 6], vertical_alignment="top")
temperature = left.slider(
    label="Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help=f"Controls randomness: a low value means less random responses."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with right.chat_message(message["role"], avatar=avatar):
        right.markdown(message["content"])

if not groq_api_key or not twitter_api_key:
    st.warning("Please enter your Groq API key and X (Twitter) Bearer Token!", icon="⚠")
else:
    os.environ['GROQ_API_KEY'] = groq_api_key
    user_input = st.chat_input("Paste an Account ID or Account ID and single Post ID")
    if user_input:
        inpt = user_input.split()
        word_count = len(inpt)
        account_name = user_input[0] if word_count > 0 else None
        tweet_id = user_input[1] if word_count > 1 else None

        client = tweepy.Client(
            bearer_token=twitter_api_key)
        if tweet_id:
            response = client.get_users_tweets(id=account_name, max_results=5, tweet_fields=['text'])
            tweet = response.data.text
        else:
            tweet = ""
            response = client.get_tweet(id=tweet_id, tweet_fields=['text'])
            tweets = response.data
            for sent in tweets:
                tweet += sent.text
        if response.status_code == 200:
            if 'eOracle'.lower() in tweet.lower():
                agent.search_knowledge = True
            else:
                agent.search_knowledge = False

            chromadb.api.client.SharedSystemClient.clear_system_cache()

            prompt_template = create_prompt()

            with right.chat_message("user", avatar='👨‍💻'):
                right.markdown("Twitter post that is between 120-160 letters long.")
            st.session_state.messages.append({"role": "user", "content": prompt_template})
            with right.chat_message("assistant"):
                stream = agent.run(prompt_template,
                                   stream=True,
                                   temperature=temperature, )
                r = generate_chat_responses(stream)
                response = right.write_stream(r)
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif response.status_code == 429:
            st.warning("Too Many Requests, Tweet Rate Limit Exceeded!", icon="⚠")
        else:
            st.warning("Error response", icon="⚠")
        chromadb.api.client.SharedSystemClient.clear_system_cache()

