import streamlit as st
from groq import Groq
from typing import Generator

st.set_page_config(
    page_icon="💬",
    page_title="Chat App",
    layout="wide",
)

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")

st.title("ChatGPT-like clone 🎈")
client = Groq(api_key=groq_api_key)


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.3-70b-versatile"

left, right = st.columns([2, 6], vertical_alignment="top")
max_tokens_range = 32768
max_tokens = left.slider(
    label="Max Tokens:",
    min_value=128,
    max_value=max_tokens_range,
    # Default value or max allowed if less
    value=min(1024, max_tokens_range),
    step=128,
    help=f"Adjust the maximum number of tokens (words) for the model's response."
)
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

if not groq_api_key:
    st.warning("Please enter your Groq API key!", icon="⚠")
else:
    prompt = st.chat_input("Say something")
    if prompt:
        with right.chat_message("user", avatar='👨‍💻'):
            right.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with right.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["groq_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )

            r = generate_chat_responses(stream)
            response = right.write_stream(r)
        st.session_state.messages.append({"role": "assistant", "content": response})

