import streamlit as st
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI


def get_response(query, chat_history):
    template = """
    あなたは優秀なアシスタントです．会話の履歴に基づき，応答してください．
    会話の履歴: {chat_history}
    質問: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI()
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"chat_history": chat_history, "user_question": query})


load_dotenv()

st.title("Streaming Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

if message := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(message)
    st.session_state.messages.append({"role": "user", "content": message})

    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(message, st.session_state.messages)
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
