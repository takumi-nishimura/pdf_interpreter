#! /usr/bin/env python
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class OpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_progress_bar = True

    def _get_len_safe_embeddings(
        self,
        texts: List[str],
        *,
        engine: str,
        chunk_size: Optional[int] = None
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

            magnitude = sum(val**2 for val in average) ** 0.5
            embeddings[i] = [val / magnitude for val in average]

        return embeddings


def main():
    load_dotenv()
    filename = "pdf_interpreter/test_en.pdf"
    pdf_doc = PyPDFLoader(filename).load()
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    docs = text_splitter.split_documents(pdf_doc)

    client = QdrantClient(":memory:")
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    collection_name = filename
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )

    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings(),
    )
    qdrant.add_documents(docs)

    llm = ChatOpenAI(
        model="llama-cpp-model",
        temperature=0.2,
        max_tokens=4048,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    retriever = qdrant.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    questions = [
        "What is this paper about?",
        "What's so great about it compared to previous research?",
        "What is the key to the technique or method?",
        "How does it validate the proposed method?",
        "Is there any discussion of this paper?",
        "What paper should I read next to better understand this paper?",
    ]
    for question in questions:
        qa(question)


if __name__ == "__main__":
    main()
