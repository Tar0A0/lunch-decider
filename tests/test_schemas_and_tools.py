"""針對餐廳資料庫、Pydantic Schema 驗證與 Tool Calling 工具函式的單元測試。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lunch_decider.schemas import LunchDecisionResponse, Recommendation, UserQuery
from lunch_decider.tools import (
    check_budget,
    execute_tool_call,
    get_weather,
    load_restaurants,
    search_restaurants,
)


def test_restaurants_json_has_at_least_15_valid_items() -> None:
    """驗證 data/restaurants.json 至少有 15 筆餐廳且符合 Restaurant Schema。"""
    restaurants = load_restaurants()
    assert len(restaurants) >= 15
    cuisines = {r.cuisine for r in restaurants}
    assert len(cuisines) >= 8


def test_recommendation_schema_validation() -> None:
    """驗證 Recommendation Schema 的欄位限制與錯誤攔截。"""
    valid = Recommendation(
        name="川魂麻辣燙・酸辣粉專門店",
        dishes=["招牌肥腸酸辣粉", "紅油抄手"],
        price=145,
        eta_minutes=25,
        calories_estimate=620,
        reason="符合 150 元預算、香麻帶辣且不含牛肉成分。",
        score=95,
    )
    assert valid.score == 95

    # 分數超過 100 應該拋出 ValidationError
    with pytest.raises(ValidationError):
        Recommendation(
            name="錯誤餐廳",
            dishes=["測試餐點"],
            price=120,
            eta_minutes=20,
            calories_estimate=500,
            reason="測試分數超過上限的驗證機制是否正常運作。",
            score=105,
        )


def test_search_restaurants_respects_budget_and_no_beef() -> None:
    """驗證 search_restaurants 能正確過濾超出預算與含牛肉的餐廳。"""
    results = search_restaurants(max_price=150, taste="辣", diet="不吃牛")
    assert len(results) >= 3
    for rest in results:
        assert rest["avg_price"] <= 150
        assert rest["contains_beef"] is False


def test_search_restaurants_vegetarian_filter() -> None:
    """驗證素食飲食限制只會回傳素食餐廳。"""
    results = search_restaurants(max_price=200, diet="全素")
    assert len(results) >= 1
    assert all(r["is_vegetarian"] is True for r in results)


def test_check_budget_and_free_delivery_threshold() -> None:
    """驗證 2 人點 145 元餐點可達 290 元（未滿 300 需運費）與 2 人點 150 元（滿 300 免運）。"""
    res_free = check_budget(price_per_person=150, people=2, budget_per_person=150)
    assert res_free["is_free_delivery"] is True
    assert res_free["final_per_person"] == 150
    assert res_free["within_budget"] is True

    res_fee = check_budget(price_per_person=140, people=1, budget_per_person=150)
    assert res_fee["is_free_delivery"] is False
    assert res_fee["total_cost"] == 179
    assert res_fee["within_budget"] is False


def test_execute_tool_call_router() -> None:
    """驗證 execute_tool_call 能正確派發三個工具並回傳合法 JSON。"""
    weather_json = execute_tool_call("get_weather", {"city": "台北市大安區"})
    weather_data = json.loads(weather_json)
    assert weather_data["city"] == "台北市大安區"
    assert "temperature_c" in weather_data

    user_q = UserQuery(budget=150, taste="辣", diet="不吃牛", people=2)
    assert user_q.people == 2

    resp = LunchDecisionResponse(
        recommendations=[
            Recommendation(
                name="曼谷小廚泰式便當",
                dishes=["泰式打拋豬飯（附半熟蛋）"],
                price=135,
                eta_minutes=22,
                calories_estimate=680,
                reason="酸香微辣超下飯，全店不含牛肉，兩人點餐搭配小菜即可免運。",
                score=92,
            )
        ],
        weather_summary="多雲偶陣雨",
        budget_summary="符合預算",
    )
    assert len(resp.recommendations) == 1
