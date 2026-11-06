"""LLM 工具呼叫 (Tool Calling) 函式庫與 JSON Schema 定義。

設計決策說明：
提供三個具實用價值的工具供 LLM 在決策過程中主動呼叫：
1. `search_restaurants`：根據預算上限、口味偏好與飲食限制（如「不吃牛」、「素食」）檢索本地餐廳資料庫。
2. `get_weather`：查詢用餐地區即時天氣（模擬資料），讓模型將氣溫與降雨納入外送時間與料理冷熱考量。
3. `check_budget`：計算多人團購的總金額、外送費分攤與滿額免運門檻，確保不超支。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lunch_decider.schemas import Restaurant

# 預設餐廳資料檔案路徑（相對於專案根目錄）
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "restaurants.json"


def load_restaurants(data_path: Path | None = None) -> list[Restaurant]:
    """讀取並以 Pydantic 驗證 `data/restaurants.json` 中的餐廳清單。"""
    target_path = data_path or DEFAULT_DATA_PATH
    raw_data = json.loads(target_path.read_text(encoding="utf-8"))
    return [Restaurant.model_validate(item) for item in raw_data]


def search_restaurants(
    max_price: int = 200,
    taste: str | None = None,
    diet: str | None = None,
    cuisine: str | None = None,
    area: str | None = None,
    data_path: Path | None = None,
) -> list[dict[str, Any]]:
    """依預算、口味偏好、飲食限制與料理類型篩選合適的餐廳候選清單。"""
    restaurants = load_restaurants(data_path)
    results: list[Restaurant] = []

    for rest in restaurants:
        # 1. 預算篩選
        if rest.avg_price > max_price:
            continue

        # 2. 飲食限制嚴格篩選（安全第一：不吃牛、不吃豬、全素／素食）
        if diet and diet not in ("無", "不限", "none", ""):
            diet_norm = diet.strip()
            if "不吃牛" in diet_norm and rest.contains_beef:
                continue
            if ("素" in diet_norm or "蔬食" in diet_norm) and not rest.is_vegetarian:
                continue
            if "不吃豬" in diet_norm and any("含豬肉" in tag for tag in rest.dietary_tags):
                continue

        # 3. 料理類型篩選（若有指定）
        if cuisine and cuisine not in ("不限", ""):
            if cuisine not in rest.cuisine and cuisine not in rest.description:
                continue

        results.append(rest)

    # 4. 根據口味與地區計算加權匹配分數以進行排序
    def match_score(r: Restaurant) -> float:
        score = r.rating * 10
        if taste and taste not in ("不限", ""):
            if taste in r.taste_tags:
                score += 25
            elif any(taste in tag or tag in taste for tag in r.taste_tags):
                score += 18
            elif taste in r.description or any(taste in d for d in r.signature_dishes):
                score += 10
        if area and area in r.city_area:
            score += 4
        return score

    results.sort(key=match_score, reverse=True)
    return [r.model_dump() for r in results[:6]]


def get_weather(city: str = "台北市大安區") -> dict[str, Any]:
    """查詢指定城市或行政區的今日午間天氣資訊（內建穩定模擬資料，不需外部金鑰）。"""
    weather_db: dict[str, dict[str, Any]] = {
        "台北市大安區": {
            "city": "台北市大安區",
            "condition": "多雲時陰偶陣雨",
            "temperature_c": 23,
            "rain_probability": 60,
            "delivery_impact": "雨天外送需求較高，建議選擇距離近、預估 25 分鐘內送達的店家，微涼天氣適合溫熱或開胃帶辣料理。",
        },
        "台北市信義區": {
            "city": "台北市信義區",
            "condition": "陰短暫陣雨",
            "temperature_c": 22,
            "rain_probability": 70,
            "delivery_impact": "商辦大樓午間雨天外送尖峰，建議優先挑選備餐快速的便當或河粉。",
        },
        "台北市中山區": {
            "city": "台北市中山區",
            "condition": "晴時多雲",
            "temperature_c": 26,
            "rain_probability": 20,
            "delivery_impact": "天氣穩定外送順暢，適合各類異國料理與清爽輕食。",
        },
    }
    for key, info in weather_db.items():
        if key in city or city in key:
            return info

    return {
        "city": city,
        "condition": "多雲舒適",
        "temperature_c": 24,
        "rain_probability": 30,
        "delivery_impact": "外送運能正常，可依個人口味自由選擇。",
    }


def check_budget(
    price_per_person: int,
    people: int = 1,
    budget_per_person: int = 150,
    delivery_fee: int = 39,
    free_delivery_threshold: int = 300,
) -> dict[str, Any]:
    """計算多人訂餐的餐點小計、外送費分攤（滿 300 免運）與每人實付金額是否符合預算。"""
    subtotal = price_per_person * people
    actual_delivery_fee = 0 if subtotal >= free_delivery_threshold else delivery_fee
    total_cost = subtotal + actual_delivery_fee
    final_per_person = round(total_cost / max(people, 1))
    within_budget = final_per_person <= budget_per_person

    return {
        "people": people,
        "subtotal": subtotal,
        "delivery_fee": actual_delivery_fee,
        "is_free_delivery": actual_delivery_fee == 0,
        "total_cost": total_cost,
        "final_per_person": final_per_person,
        "budget_per_person": budget_per_person,
        "within_budget": within_budget,
        "summary": (
            f"{people} 人訂餐小計 ${subtotal}，"
            f"{'已達滿 $300 免外送費門檻' if actual_delivery_fee == 0 else f'外送費 ${actual_delivery_fee}'}，"
            f"總計 ${total_cost}（平均每人 ${final_per_person}，"
            f"{'符合' if within_budget else '超出'}預算上限 ${budget_per_person}）。"
        ),
    }


# 供 Ollama / OpenAI 相容 API 使用的標準 Tool Calling JSON Schema
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "根據每人預算上限、口味偏好與飲食限制搜尋符合條件的本地外送餐廳。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "integer",
                        "description": "每人最高預算金額（新台幣）",
                    },
                    "taste": {
                        "type": "string",
                        "description": "口味偏好關鍵字，例如：辣、酸辣、清爽、濃郁",
                    },
                    "diet": {
                        "type": "string",
                        "description": "飲食限制，例如：不吃牛、全素、不吃豬",
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "料理類型，例如：泰式料理、川味中式、健康輕食",
                    },
                    "area": {
                        "type": "string",
                        "description": "用餐所在行政區，例如：台北市大安區",
                    },
                },
                "required": ["max_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查詢用餐所在地區的即時天氣與外送交通影響評估。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市與行政區名稱，例如：台北市大安區",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_budget",
            "description": "試算多人外送訂單含運費分攤後的總金額與每人平均花費是否符合預算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "price_per_person": {
                        "type": "integer",
                        "description": "預估單人餐點價格（新台幣）",
                    },
                    "people": {
                        "type": "integer",
                        "description": "用餐人數",
                    },
                    "budget_per_person": {
                        "type": "integer",
                        "description": "使用者設定的每人預算上限",
                    },
                },
                "required": ["price_per_person", "people", "budget_per_person"],
            },
        },
    },
]


def execute_tool_call(func_name: str, arguments: dict[str, Any]) -> str:
    """根據 LLM 回傳的工具名稱與參數執行對應 Python 函式，並回傳 JSON 字串。"""
    if func_name == "search_restaurants":
        res = search_restaurants(
            max_price=int(arguments.get("max_price", 200)),
            taste=arguments.get("taste"),
            diet=arguments.get("diet"),
            cuisine=arguments.get("cuisine"),
            area=arguments.get("area"),
        )
        return json.dumps(res, ensure_ascii=False)

    if func_name == "get_weather":
        res = get_weather(city=str(arguments.get("city", "台北市大安區")))
        return json.dumps(res, ensure_ascii=False)

    if func_name == "check_budget":
        res = check_budget(
            price_per_person=int(arguments.get("price_per_person", 150)),
            people=int(arguments.get("people", 1)),
            budget_per_person=int(arguments.get("budget_per_person", 150)),
        )
        return json.dumps(res, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {func_name}"}, ensure_ascii=False)
