"""LLM and Embedding provider utility supporting Ollama (llama3.2:latest), OpenAI, Gemini, and deterministic fallback."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Generator
from app.utilities.config import settings

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "as", "into", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "and", "or", "but", "if", "while", "because", "how", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "do", "does",
    "did", "have", "has", "had", "can", "could", "will", "would", "should",
    "tell", "me", "please", "give", "show", "find",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_local_semantic_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Deterministic multi-resolution feature hashing embedding (word unigrams,
    word stems/prefixes, bigrams, and character trigrams) normalized to unit L2 norm.
    """
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower()) or ["empty"]

    counts = Counter(tokens)

    def _add_feature(feature: str, weight: float) -> None:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        for slot in range(4):
            idx = int.from_bytes(digest[slot * 4 : slot * 4 + 2], "big") % dim
            sign = 1.0 if (digest[slot * 4 + 2] % 2 == 0) else -1.0
            vec[idx] += sign * weight

    for token, freq in counts.items():
        tf_weight = 1.0 + math.log1p(freq)
        _add_feature(f"tok:{token}", tf_weight * 3.0)
        if len(token) >= 4:
            _add_feature(f"stem4:{token[:4]}", tf_weight * 2.0)
        if len(token) >= 5:
            _add_feature(f"stem5:{token[:5]}", tf_weight * 2.2)
        padded = f"<{token}>"
        for i in range(len(padded) - 2):
            _add_feature(f"c3:{padded[i:i+3]}", 0.35)

    for i in range(len(tokens) - 1):
        _add_feature(f"bi:{tokens[i]}_{tokens[i+1]}", 1.8)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        vec[0] = 1.0
        return vec
    return [round(v / norm, 6) for v in vec]


def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding of dimension settings.embedding_dimension (default 384)."""
    dim = settings.embedding_dimension
    provider = settings.llm_provider.lower()

    if provider in {"openai"} and settings.openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings

            emb_client = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=dim,
            )
            return emb_client.embed_query(text)
        except Exception:
            pass

    return compute_local_semantic_embedding(text, dim=dim)


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for a batch of texts."""
    dim = settings.embedding_dimension
    provider = settings.llm_provider.lower()

    if provider in {"openai"} and settings.openai_api_key and texts:
        try:
            from langchain_openai import OpenAIEmbeddings

            emb_client = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=dim,
            )
            return emb_client.embed_documents(texts)
        except Exception:
            pass

    return [compute_local_semantic_embedding(t, dim=dim) for t in texts]


def generate_llm_response(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """
    Invoke the configured Chat LLM synchronously with (system_prompt, user_prompt).
    Returns (response_text, active_provider_name).
    """
    provider = settings.llm_provider.lower()

    # 1. Local Ollama (default small model: llama3.2:latest)
    if provider in {"ollama", "auto"}:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.2,
            )
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = str(resp.content).strip()
            if content:
                return content, f"ollama:{settings.ollama_model}"
        except Exception:
            pass

    # 2. OpenAI
    if provider in {"openai", "auto"} and settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.2,
            )
            resp = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return str(resp.content).strip(), f"openai:{settings.openai_model}"
        except Exception:
            pass

    # 3. Google Gemini
    if provider in {"gemini", "auto"} and settings.gemini_api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.gemini_api_key)
            resp = client.models.generate_content(
                model=settings.gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                ),
            )
            if resp and resp.text:
                return resp.text.strip(), f"gemini:{settings.gemini_model}"
        except Exception:
            pass

    return "", "local_rag_synthesizer"


def stream_llm_response(
    system_prompt: str, user_prompt: str
) -> Generator[tuple[str, str], None, None]:
    """
    Stream tokens word-by-word from Ollama (llama3.2:latest) or OpenAI.
    Yields (token_text, provider_name).
    """
    provider = settings.llm_provider.lower()

    # 1. Stream directly from Ollama small model (llama3.2:latest)
    if provider in {"ollama", "auto"}:
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.2,
            )
            yielded_any = False
            provider_tag = f"ollama:{settings.ollama_model}"
            for chunk in llm.stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ):
                token = str(chunk.content or "")
                if token:
                    yielded_any = True
                    yield token, provider_tag
            if yielded_any:
                return
        except Exception:
            pass

    # 2. Stream from OpenAI if configured
    if provider in {"openai", "auto"} and settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=0.2,
            )
            yielded_any = False
            provider_tag = f"openai:{settings.openai_model}"
            for chunk in llm.stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            ):
                token = str(chunk.content or "")
                if token:
                    yielded_any = True
                    yield token, provider_tag
            if yielded_any:
                return
        except Exception:
            pass
