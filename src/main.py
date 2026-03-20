import os
import asyncio
import time

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from pydantic import BaseModel, Field, field_validator
import nltk
import faiss
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.functional import embedding

from src.config.project_config import *

import json
import string
from typing import Dict, List, Optional, Tuple


FAISS_NUM_CANDIDATES = 150
FINAL_NUM_RESULTS = 10
MAX_LEN_EMBEDDING_KNRM = 30
DetectorFactory.seed = 0


class EnglishOnly:
    async def __call__(self, text: str):
        if not text or len(text) < 10:
            return True

        try:
            lang = await asyncio.to_thread(detect, text)

            return lang == 'en'

        except LangDetectException:
            return False


english_only = EnglishOnly()

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        kernel_output = torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
        return kernel_output


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = []):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        step_mu =  1.0 / (self.kernel_num - 1)
        mus = torch.linspace(-1 + step_mu, 1, steps=self.kernel_num)
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num - 1):
            kernels.append(GaussianKernel(mu=mus[i].item(), sigma=self.sigma))
        kernels.append(GaussianKernel(mu=mus[-1].item(), sigma=self.exact_sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        layers = []
        input_dim = self.kernel_num
        for hidden_dim in self.out_layers:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim

        layers.append(torch.nn.Linear(input_dim, 1))
        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        query_emb = self.embeddings(query)
        doc_emb = self.embeddings(doc)
        query_norm = F.normalize(query_emb, p=2, dim=-1)
        doc_norm = F.normalize(doc_emb, p=2, dim=-1)
        matching_matrix = torch.matmul(query_norm, doc_norm.transpose(1, 2))
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class GloveVectorizer:
    def __init__(self, glove_embeddings: str = EMB_PATH_GLOVE):

        self.glove = dict()
        with open(EMB_PATH_GLOVE, 'r', encoding='utf-8') as file:
            for line in file:
                word, *embeddings = line.strip().split()
                embeddings = list(map(float, embeddings))
                self.glove[word] = embeddings


        print(f"Load GloVe dictionary: {len(self.glove)} embeddings")
        self.dimension = len(self.glove.get('0'))

        # Загружаем NLTK данные
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        self.punct_table = str.maketrans({p: ' ' for p in string.punctuation})

    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Убираем пунктуацию
        text = text.translate(self.punct_table)
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        # Токенизация
        tokens = nltk.word_tokenize(text)
        return tokens

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Преобразует текст в вектор, используя только слова из GLOVE

        Args:
            text: входной текст

        Returns:
            вектор размерности embedding_dim
        """
        tokens = self.tokenize(text)

        if not tokens:
            return np.zeros(self.dimension)

        vectors = []

        # Берем только слова, которые есть в GLOVE
        for token in tokens:
            if token in self.glove.keys():
                vectors.append(self.glove.get(token))

        if not vectors:
            # Если ни одного слова нет в GLOVE
            return np.zeros(self.dimension)

        # Усредняем векторы
        result = np.mean(vectors, axis=0)

        # Нормализуем
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

        return result.astype(np.float32)

    def texts_to_vectors(self, texts: List[str]) -> np.ndarray:
        """
        Преобразует список текстов в матрицу векторов
        """
        vectors = []
        for text in texts:
            vec = self.text_to_vector(text)
            vectors.append(vec)

        return np.array(vectors)

    def get_coverage(self, text: str) -> dict:
        """
        Возвращает статистику покрытия текста словами из GLOVE
        """
        tokens = self.tokenize(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return {
                'total_tokens': 0,
                'glove_tokens': 0,
                'coverage': 0.0,
                'missing_words': []
            }

        glove_tokens = [t for t in tokens if t in self.glove.keys()]
        missing_tokens = [t for t in tokens if t not in self.glove.keys()]

        return {
            'total_tokens': total_tokens,
            'glove_tokens': len(glove_tokens),
            'coverage': len(glove_tokens) / total_tokens,
            'missing_words': missing_tokens[:10]
        }

class Searcher:
    def __init__(
            self,
            vectorizer: GloveVectorizer,
            path_vocab: str = VOCAB_PATH,
            path_knrm: str = EMB_PATH_KNRM,
            path_mlp_weights: str = MLP_PATH
        ):

        self.vectorizer = vectorizer
        self.dimension = vectorizer.dimension
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, str] = {}
        self.doc_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.documents_idx: Dict[str, str] = {}

        with open(path_vocab, 'rb') as f:
            self.vocab = json.load(f)

        print(f"Load vocab: {len(self.vocab)}")

        # Проверка специальных токенов
        print(f"  PAD token: {self.vocab.get('PAD', 'not found')}")
        print(f"  OOV token: {self.vocab.get('OOV', 'not found')}")
        print(f"  UNK token: {self.vocab.get('UNK', 'not found')}")

        # Примеры слов из словаря
        sample_words = list(self.vocab.keys())[:10]
        print(f"  Примеры слов: {sample_words}")

        self.path_knrm = path_knrm
        self.path_mlp_weights = path_mlp_weights
        self.model = self.build_knrm()
        self.model.eval()

    def build_knrm(self) -> KNRM:
        with open(self.path_knrm, 'rb') as f:
            emb_matrix = pickle.load(f)

        print(f"Load embedding matrix: {emb_matrix.shape}")

        with open(self.path_mlp_weights, 'rb') as f:
            mlp_weights = pickle.load(f)

        print(f"Load dictionary MLP weights: {mlp_weights.keys()}")

        model = KNRM(emb_matrix, freeze_embeddings=False)

        # Загружаем веса
        state_dict = model.state_dict()
        for name, param in mlp_weights.items():
            if name in state_dict:
                state_dict[name].copy_(torch.FloatTensor(param))
                print(f"  Загружен слой: {name}")

        model.load_state_dict(state_dict)
        return model

    def build_index(self, documents: Dict[str, str]):
        """
        Строим FAISS индекс из документов

        Args:
             documents: словарь {id: текст}
        """
        print(f"building index for {len(documents)} documents...")

        self.documents = documents
        self.doc_ids = list(documents.keys())

        self.documents_idx = {value: key for key, value in self.documents.items()}

        vectors = []
        failed_docs = []

        for i, (doc_id, text) in enumerate(documents.items()):
            vec = self.vectorizer.text_to_vector(text)

            if np.any(vec):
                vectors.append(vec)
            else:
                failed_docs.append(doc_id)


        print(f" Processed {len(documents)} documents")

        if not vectors:
            raise ValueError("There are no documents for indexing.")

        self.vectors = np.array(vectors, dtype=np.float32)
        print(f" Matrix of vectors {self.vectors.shape}")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.vectors)

        print(f"The index has been created! Size: {self.index.ntotal}")

        if failed_docs:
            print(f"Failed to vectorize {len(failed_docs)} documents")

        index_size = self.index.ntotal
        return index_size

    def search(self, query: str, k: int = FAISS_NUM_CANDIDATES) -> List[str]:
        if self.index is None:
            raise ValueError("Индекс не построен")

        query_vector = self.vectorizer.text_to_vector(query)

        # Проверяем, есть ли слова из GLOVE в запросе
        coverage = self.vectorizer.get_coverage(query)
        if coverage['coverage'] == 0:
            return []  # Нет слов из GLOVE

        query_vector = query_vector.reshape(1, -1)

        _, indices = self.index.search(query_vector, k)

        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.doc_ids):
                doc_text = self.documents[self.doc_ids[idx]]
                results.append(doc_text)

        return results

    def get_word_embeddings_idx(self, text: str) -> torch.LongTensor:
        tokens = self.vectorizer.tokenize(text)
        embeddings_idx = []

        for token in tokens:
            idx = self.vocab.get(token, 1)
            embeddings_idx.append(idx)

        if len(embeddings_idx) > MAX_LEN_EMBEDDING_KNRM:
            embeddings_idx = embeddings_idx[:MAX_LEN_EMBEDDING_KNRM]
        else:
            embeddings_idx = embeddings_idx + [0] * (MAX_LEN_EMBEDDING_KNRM - len(embeddings_idx))

        return torch.LongTensor(embeddings_idx)

    def rerank_with_knrm(self, query: str, candidate_texts: List[str], top_k: int = FINAL_NUM_RESULTS) -> List[
        Tuple[str, float]]:
        """
        Переранжирует кандидатов с помощью KNRM
        """
        if not candidate_texts:
            return []

        # Получаем эмбеддинги запроса
        query_emb = self.get_word_embeddings_idx(query).unsqueeze(0)

        candidates_with_scores = []

        with torch.no_grad():  # Отключаем градиенты для инференса
            for text in candidate_texts:
                doc_emb = self.get_word_embeddings_idx(text).unsqueeze(0)

                # Подготавливаем входные данные для модели
                inputs = {
                    'query': query_emb,
                    'document': doc_emb
                }

                # Получаем оценку от KNRM
                score = self.model.predict(inputs).item()
                candidates_with_scores.append((text, score))

        # Сортируем по убыванию оценки
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

        return candidates_with_scores[:top_k]


class UpdateIndexRequest(BaseModel):
    documents: Dict[str, str] = Field(..., description="Dictionary of documents")

    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            raise ValueError('Documents dictionary cannot be empty')

        if len(v) > 10_000_000:
            raise ValueError(f'Too many documents: {len(v)} (max 10_000_000)')

        return v


class UpdateIndexResponse(BaseModel):
    status: str = Field(..., description="Operation status (ok/error)")
    index_size: int = Field(..., description="Size of indexes after update")

    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ['ok', 'error']:
            raise ValueError('Status must be ok or error')
        return v


class QueryRequest(BaseModel):
    queries: List[str] = Field(..., description="Dictionary of queries")


vectorizer: Optional[GloveVectorizer] = None
searcher: Optional[Searcher] = None
initialized_components = False

async def initialize_components():
    """Фоновая инициализация компонентов"""
    global vectorizer, searcher, initialized_components

    try:
        loop = asyncio.get_event_loop()
        vectorizer = await loop.run_in_executor(None, GloveVectorizer)
        searcher = await loop.run_in_executor(None, Searcher, vectorizer)
        initialized_components = True

    except Exception as e:
        initialized_components = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(initialize_components())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def post_query(request: QueryRequest):
    global searcher

    if searcher is None or searcher.index is None or searcher.index.ntotal == 0:
        return {
            "status": "FAISS is not initialized"
        }

    try:
        results = []

        for query in request.queries:
            is_english = await english_only(query)

            if not is_english:
                results.append({
                    'query': query,
                    'lang_check': False,
                    'suggestions': []
                })
                continue

            candidates = searcher.search(query, k=FAISS_NUM_CANDIDATES)

            if len(candidates) == 0:
                results.append({
                    'query': query,
                    'lang_check': True,
                    'suggestions': []
                })
                continue

            reranked = searcher.rerank_with_knrm(query, candidates, top_k=FINAL_NUM_RESULTS)

            suggestions = []
            for text, _ in reranked:
                if text in searcher.documents_idx.keys():
                    doc_id = searcher.documents_idx[text]
                    suggestions.append((doc_id, text))

            results.append({
                'query': query,
                'lang_check': True,
                'suggestions': suggestions
            })

        return {
            'status': 'ok',
            'results': results
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Search failed",
                "message": str(e)
            }
        )

@app.post("/update_index", response_model=UpdateIndexResponse)
async def update_index(request: UpdateIndexRequest):
    global searcher, initialized_components
    try:
        print(f"A request to update the index was received")
        print(f"Documents: {len(request.documents)}")

        if initialized_components is False:
            return {
                "status": "error",
                "message": "Searcher no initialized",
            }

        index_size = await asyncio.to_thread(
            searcher.build_index,
            request.documents
        )

        return {
            "status": "ok",
            "index_size": index_size
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Building index failed",
                "message": str(e)
            }
        )

@app.get("/ping")
async def ping():
    global initialized_components

    if initialized_components:
        return {
            "status": "ok",
            "message": "Service ready"
        }
    else:
        return {
            "status": "error",
            "message": "Vectorizer or Searcher not initialized"
        }

if __name__ == "__main__":
    uvicorn.run('main:app', host="127.0.0.1", port=11000, reload=True)
