"""Embedding 向量語意相似度檢索模組（作業第 4 項 LLM 能力／加分項）。

設計決策說明：
1. 將使用者的自然語言需求（口味、飲食限制、預算與地區）與 `data/restaurants.json` 中每間餐廳的
   料理類型、招牌菜與特色描述轉為數值向量，透過餘弦相似度 (Cosine Similarity) 進行語意排序。
2. 支援雙模式向量化：
   - 遠端／本地模型模式 (`use_ollama_embed=True`)：呼叫 Ollama `/api/embed` 端點取得深度學習向量。
   - 內建中文字符 N-gram 向量模式 (預設保底與 `--mock` 模式)：使用針對繁體中文餐飲詞彙最佳化的
     Unigram + Bigram 特徵雜湊向量，不需額外下載數百 MB 模型即可在毫秒內算出精準的餘弦相似度。
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any

import httpx

from lunch_decider.schemas import Restaurant, UserQuery
from lunch_decider.tools import load_restaurants


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """計算兩個向量的餘弦相似度 (Cosine Similarity)，回傳值介於 0.0 ~ 1.0。"""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _restaurant_to_document(rest: Restaurant) -> str:
    """將餐廳結構化欄位拼接為適合 Embedding 檢索的語意文件字串。"""
    beef_tag = "含牛肉" if rest.contains_beef else "不吃牛可 無牛肉"
    veg_tag = "全素 蔬食 素食" if rest.is_vegetarian else "葷食"
    return (
        f"{rest.name} {rest.cuisine} {rest.city_area} "
        f"口味：{' '.join(rest.taste_tags)} "
        f"飲食標籤：{' '.join(rest.dietary_tags)} {beef_tag} {veg_tag} "
        f"招牌菜：{' '.join(rest.signature_dishes)} "
        f"價格：{rest.avg_price}元 說明：{rest.description}"
    )


def _query_to_text(query: UserQuery) -> str:
    """將使用者查詢條件轉為語意查詢字串。"""
    return (
        f"我想在 {query.area} 點午餐外送，口味偏好：{query.taste}，"
        f"飲食限制：{query.diet}，預算每人 {query.budget} 元以內。"
    )


def _char_ngram_vectors(texts: list[str], dim: int = 256) -> list[list[float]]:
    """將多筆繁體中文文字轉為固定維度的字元 Unigram + Bigram TF-IDF 加權雜湊向量。"""
    tokenized_docs: list[Counter[str]] = []
    doc_freq: Counter[str] = Counter()

    for text in texts:
        cleaned = "".join(ch for ch in text.lower() if not ch.isspace())
        unigrams = list(cleaned)
        bigrams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
        counts = Counter(unigrams + bigrams)
        tokenized_docs.append(counts)
        for token in counts:
            doc_freq[token] += 1

    num_docs = len(texts)
    vectors: list[list[float]] = []
    for counts in tokenized_docs:
        vec = [0.0] * dim
        for token, tf in counts.items():
            idf = math.log((num_docs + 1) / (doc_freq[token] + 1)) + 1.0
            # 使用穩定雜湊對應至固定維度槽位
            idx = sum(ord(c) * (i + 31) for i, c in enumerate(token)) % dim
            vec[idx] += tf * idf
        vectors.append(vec)
    return vectors


def _fetch_ollama_embeddings(texts: list[str]) -> list[list[float]] | None:
    """嘗試呼叫本地 Ollama `/api/embed` 取得向量；若服務未啟動則回傳 None 由本地向量器接手。"""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{base_url}/api/embed",
                json={"model": embed_model, "input": texts},
            )
            if resp.status_code == 200:
                embeddings = resp.json().get("embeddings")
                if isinstance(embeddings, list) and len(embeddings) == len(texts):
                    return embeddings
    except Exception:
        return None
    return None


def rank_restaurants_by_embedding(
    query: UserQuery,
    top_k: int = 5,
    use_ollama_embed: bool = False,
) -> list[dict[str, Any]]:
    """計算使用者需求與所有餐廳的 Embedding 餘弦相似度，回傳前 top_k 名候選餐廳。"""
    restaurants = load_restaurants()

    # 先過濾掉嚴格違反飲食限制（例如不吃牛卻含牛肉）或嚴重超支的餐廳
    eligible: list[Restaurant] = []
    for r in restaurants:
        if "不吃牛" in query.diet and r.contains_beef:
            continue
        if ("素" in query.diet or "蔬食" in query.diet) and not r.is_vegetarian:
            continue
        if r.avg_price > query.budget + 30:
            continue
        eligible.append(r)

    if not eligible:
        eligible = restaurants

    query_text = _query_to_text(query)
    doc_texts = [_restaurant_to_document(r) for r in eligible]
    all_texts = [query_text, *doc_texts]

    vectors = _fetch_ollama_embeddings(all_texts) if use_ollama_embed else None
    embed_source = "ollama-embed" if vectors is not None else "local-char-ngram-256d"
    if vectors is None:
        vectors = _char_ngram_vectors(all_texts, dim=256)

    query_vec = vectors[0]
    doc_vecs = vectors[1:]

    scored: list[dict[str, Any]] = []
    for rest, d_vec in zip(eligible, doc_vecs, strict=False):
        sim = cosine_similarity(query_vec, d_vec)
        scored.append(
            {
                "id": rest.id,
                "name": rest.name,
                "cuisine": rest.cuisine,
                "avg_price": rest.avg_price,
                "similarity": round(sim, 4),
                "embed_engine": embed_source,
            }
        )

    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:top_k]
