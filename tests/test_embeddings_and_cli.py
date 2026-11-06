"""針對 Embedding 向量檢索與 CLI 完整執行流程的單元測試。"""

from __future__ import annotations

import json
from pathlib import Path

from lunch_decider.cli import run_cli
from lunch_decider.embeddings import cosine_similarity, rank_restaurants_by_embedding
from lunch_decider.schemas import UserQuery


def test_cosine_similarity_math() -> None:
    """驗證餘弦相似度數學計算正確性。"""
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    assert cosine_similarity([], []) == 0.0


def test_rank_restaurants_by_embedding_filters_and_ranks() -> None:
    """驗證 Embedding 檢索能過濾牛肉餐廳並按相似度由高到低排序。"""
    query = UserQuery(budget=150, taste="辣", diet="不吃牛", people=2, area="台北市大安區")
    candidates = rank_restaurants_by_embedding(query, top_k=5, use_ollama_embed=False)

    assert len(candidates) == 5
    # 確認按相似度降冪排列
    similarities = [c["similarity"] for c in candidates]
    assert similarities == sorted(similarities, reverse=True)
    # 確認老張記牛肉麵不在候選清單中
    assert all("老張記" not in c["name"] for c in candidates)


def test_cli_end_to_end_mock_mode(tmp_path: Path) -> None:
    """驗證 CLI --mock 模式能完整跑完四大能力並輸出 JSON 與 Markdown 報告。"""
    out_dir = tmp_path / "results"
    result = run_cli(
        [
            "--budget",
            "150",
            "--taste",
            "辣",
            "--diet",
            "不吃牛",
            "--people",
            "2",
            "--mock",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert len(result.response.recommendations) == 3
    assert len(result.embedding_candidates) == 5
    assert len(result.tool_calls_log) == 3
    assert "AI 外送決策主廚即時講評" in result.streamed_commentary

    json_file = out_dir / "latest_decision.json"
    md_file = out_dir / "latest_decision.md"
    assert json_file.exists()
    assert md_file.exists()

    saved_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert saved_data["query"]["budget"] == 150
    assert saved_data["query"]["diet"] == "不吃牛"
    assert len(saved_data["response"]["recommendations"]) == 3
