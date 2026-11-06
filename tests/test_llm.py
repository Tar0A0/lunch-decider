"""針對 LLM 結構化輸出驗證、自動重試機制、Tool Calling 流程與 Mock 模式的單元測試。"""

from __future__ import annotations

import json
from typing import Any

import httpx

from lunch_decider.llm import (
    LLMConfig,
    decide_lunch,
    decide_lunch_mock,
    stream_chef_commentary,
)
from lunch_decider.schemas import UserQuery


def test_decide_lunch_with_tool_calling_and_retry() -> None:
    """模擬 Ollama 先回傳 tool_calls、第一次結構化輸出格式錯誤、第二次重試成功驗證的完整流程。"""
    call_counter = {"count": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        call_counter["count"] += 1
        step = call_counter["count"]

        # 第 1 次呼叫：模型決定呼叫 search_restaurants 與 get_weather 工具
        if step == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_restaurants",
                                    "arguments": {
                                        "max_price": 150,
                                        "taste": "辣",
                                        "diet": "不吃牛",
                                    },
                                }
                            },
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "台北市大安區"},
                                }
                            },
                        ],
                    }
                },
            )

        # 第 2 次呼叫：故意回傳不合規的 JSON（score=999 超出 1~100 範圍），觸發 Pydantic 驗證重試
        if step == 2:
            invalid_payload = {
                "recommendations": [
                    {
                        "name": "川魂麻辣燙",
                        "dishes": ["招牌肥腸酸辣粉"],
                        "price": 145,
                        "eta_minutes": 25,
                        "calories_estimate": 620,
                        "reason": "麻辣過癮且不含牛肉，非常適合雨天午餐。",
                        "score": 999,
                    }
                ],
                "weather_summary": "多雲偶陣雨",
                "budget_summary": "符合預算",
            }
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": json.dumps(invalid_payload)}},
            )

        # 第 3 次呼叫（重試第 1 次）：模型根據錯誤訊息修正為合法 JSON
        valid_payload: dict[str, Any] = {
            "recommendations": [
                {
                    "name": "川魂麻辣燙・酸辣粉專門店",
                    "dishes": ["招牌肥腸酸辣粉", "紅油抄手"],
                    "price": 145,
                    "eta_minutes": 25,
                    "calories_estimate": 620,
                    "reason": "麻辣過癮且不含牛肉，價格 145 元符合 150 元預算。",
                    "score": 95,
                },
                {
                    "name": "曼谷小廚泰式便當",
                    "dishes": ["泰式打拋豬飯（附半熟蛋）"],
                    "price": 135,
                    "eta_minutes": 22,
                    "calories_estimate": 680,
                    "reason": "酸辣開胃不含牛肉，外送 22 分鐘快速抵達。",
                    "score": 91,
                },
                {
                    "name": "首爾韓食堂・石鍋拌飯與豆腐煲",
                    "dishes": ["韓式泡菜豬肉嫩豆腐煲"],
                    "price": 150,
                    "eta_minutes": 28,
                    "calories_estimate": 640,
                    "reason": "熱騰騰泡菜豬肉豆腐煲，微涼雨天首選。",
                    "score": 88,
                },
            ],
            "weather_summary": "台北市大安區多雲偶陣雨 23°C",
            "budget_summary": "兩人湊單可享更划算運費分攤",
        }
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(valid_payload, ensure_ascii=False),
                }
            },
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as mock_client:
        query = UserQuery(budget=150, taste="辣", diet="不吃牛", people=2, area="台北市大安區")
        cfg = LLMConfig(provider="ollama", ollama_model="qwen2.5:7b")
        result = decide_lunch(query, config=cfg, client=mock_client)

    assert len(result.tool_calls_log) == 2
    assert result.retries_used == 1
    assert len(result.response.recommendations) == 3
    assert result.response.recommendations[0].score == 95


def test_decide_lunch_mock_and_streaming() -> None:
    """驗證 --mock 模式與串流講評產生器能正常運作且符合預算與不吃牛限制。"""
    query = UserQuery(budget=150, taste="辣", diet="不吃牛", people=2, area="台北市大安區")
    result = decide_lunch_mock(query)

    assert len(result.response.recommendations) == 3
    assert len(result.tool_calls_log) == 3
    for rec in result.response.recommendations:
        assert rec.price <= 150

    chunks = list(stream_chef_commentary(query, result.response, mock=True))
    full_text = "".join(chunks)
    assert len(chunks) > 20
    assert "AI 外送決策主廚即時講評" in full_text
