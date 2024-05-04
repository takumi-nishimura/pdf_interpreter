from typing import Iterable, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


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
            ### openai.resources.embeddings.Embeddings.create
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
                print("create empty")
                average_embedded = self.client.create(
                    input="", **self._invocation_params
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                average = average_embedded["data"][0]["embedding"]
            else:
                print("create average")
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


load_dotenv()
text = "What is the capital of Japan?"
embed_model = OpenAIEmbeddings(openai_api_base="http://localhost:9877/v1")
embeddings = embed_model.embed_documents([text])
print(f"embed: {embeddings}")
