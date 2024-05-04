import io
import os
import threading
from queue import Queue
from typing import Any, Iterable, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

OPENAI_API_MODEL = os.environ.get("OPENAI_API_MODEL")


class OpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_progress_bar = True

    def _get_len_safe_embeddings(
        self,
        texts: List[str],
        *,
        engine: str,
        chunk_size: Optional[int] = None,
    ) -> List[List[float]]:

        chunk_size = 30
        chunks = []
        indices = []

        for i, text in enumerate(texts):
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i : i + chunk_size]))

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(chunks), chunk_size))
            except ImportError:
                _iter = range(0, len(chunks), chunk_size)
        else:
            _iter = range(0, len(chunks), chunk_size)

        batched_embeddings: List[List[float]] = []
        for i in _iter:
            response = self.client.create(
                input=chunks[i : i + 1], **self._invocation_params
            )
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(chunks[i]))

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average_embedded = self.client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                average = average_embedded["data"][0]["embedding"]
            else:
                total_weight = sum(num_tokens_in_batch[i])
                average = [
                    sum(
                        val * weight
                        for val, weight in zip(
                            embedding, num_tokens_in_batch[i]
                        )
                    )
                    / total_weight
                    for embedding in zip(*_result)
                ]

            magnitude = sum(val**2 for val in average[0]) ** 0.5
            embeddings[i] = [val / magnitude for val in average[0]]

        return embeddings


class StreamQueueHandler(BaseCallbackHandler):
    def __init__(self):
        self.stream_queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs: Any):
        self.stream_queue.put({"event": "new_token", "token": token})

    def on_llm_end(self, *args, **kwargs):
        self.stream_queue.put({"event": "end", "token": ""})


def load_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()
    with io.BytesIO(response.content) as open_pdf_file:
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(open_pdf_file):
            interpreter.process_page(page)
            text = retstr.getvalue()
        device.close()
        retstr.close()
    return text, response.headers["ETag"]


def qdrant_client(texts, name):
    _client = QdrantClient(":memory:")
    _collections = _client.get_collections().collections
    _collection_names = [_collection.name for _collection in _collections]
    _collection_name = name
    if _collection_name not in _collection_names:
        _client.create_collection(
            collection_name=_collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
    _qdrant = Qdrant(
        client=_client,
        collection_name=_collection_name,
        embeddings=OpenAIEmbeddings(),
    )
    _qdrant.add_texts(texts)
    return _qdrant.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )


@st.cache_resource
def load_embeddings(file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128
    )
    pdf_text = extract_text(file)
    texts = text_splitter.split_text(pdf_text)
    retriever = qdrant_client(texts, pdf_file.name)
    return retriever


def get_response(query, chat_history, retriever):
    template = """
    あなたは優秀なアシスタントです．会話の履歴に基づき，800字程度で応答してください．
    会話の履歴: {chat_history}
    質問: {user_question}
    """

    if not retriever:
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            prompt
            | ChatOpenAI(streaming=True, model=OPENAI_API_MODEL)
            | StrOutputParser()
        )
        return chain.stream(
            {"chat_history": chat_history, "user_question": query}
        )
    else:
        handler = StreamQueueHandler()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                streaming=True, callbacks=[handler], model=OPENAI_API_MODEL
            ),
            chain_type="stuff",
            retriever=retriever,
        )

        threading.Thread(
            target=qa,
            args=({"query": query, "chat_history": chat_history},),
        ).start()

        def qa_stream():
            _stream = {"event": "not_yet", "token": ""}
            while not _stream["event"] == "end":
                _stream = handler.stream_queue.get()
                yield _stream["token"]

        return qa_stream()


load_dotenv()

st.set_page_config(layout="wide")
st.title("PDF Interpreter")
emb_retriever = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

option = st.radio(
    "Choose an option:", ("Enter a PDF URL", "Upload a PDF file")
)
if option == "Enter a PDF URL":
    pdf_url = st.text_input("Enter PDF URL here.")
    if pdf_url:
        pdf_text, name = load_pdf_from_url(pdf_url)
elif option == "Upload a PDF file":
    pdf_file = st.file_uploader("Upload PDF here.", type=["pdf"])
    if pdf_file:
        emb_retriever = load_embeddings(pdf_file)
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(prompt, st.session_state["messages"], emb_retriever)
        )
    st.session_state["messages"].append(
        {"role": "assistant", "content": response}
    )
