"""Pydantic 資料模型與結構化輸出 Schema 定義。

設計決策說明：
1. 依照作業規範，定義 `Recommendation` schema，包含：
   name, dishes, price, eta_minutes, calories_estimate, reason, score 七個核心欄位。
2. 利用 Pydantic Field 的 ge / le / min_length 等限制，在解析 LLM 回傳的 JSON 時進行嚴格資料驗證，
   若模型輸出不合規（如分數超過 100 或價格為負數），會拋出 ValidationError 觸發自動重試機制。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Restaurant(BaseModel):
    """本地餐廳資料庫 (`data/restaurants.json`) 的單筆資料結構。"""

    id: str = Field(..., description="餐廳唯一識別碼")
    name: str = Field(..., min_length=1, description="餐廳名稱")
    cuisine: str = Field(..., description="料理類型，例如：泰式料理、川味中式")
    city_area: str = Field(..., description="所在城市與行政區")
    taste_tags: list[str] = Field(default_factory=list, description="口味標籤，例如：辣、清爽")
    dietary_tags: list[str] = Field(default_factory=list, description="飲食限制標籤")
    contains_beef: bool = Field(default=False, description="餐點是否含有牛肉成分")
    is_vegetarian: bool = Field(default=False, description="是否為素食／蔬食餐廳")
    avg_price: int = Field(..., gt=0, description="平均每人消費金額（新台幣）")
    eta_minutes: int = Field(..., gt=0, le=120, description="預估外送抵達時間（分鐘）")
    rating: float = Field(..., ge=0.0, le=5.0, description="外送平台評分（0~5）")
    calories_estimate: int = Field(..., gt=0, description="平均每份餐點預估熱量（大卡 kcal）")
    signature_dishes: list[str] = Field(..., min_length=1, description="招牌推薦餐點清單")
    description: str = Field(..., description="餐廳特色描述")


class UserQuery(BaseModel):
    """使用者從 CLI 輸入的午餐決策條件。"""

    budget: int = Field(default=150, gt=0, description="每人預算上限（新台幣）")
    taste: str = Field(default="不限", description="口味偏好，例如：辣、酸辣、清爽、濃郁")
    diet: str = Field(default="無", description="飲食限制，例如：不吃牛、全素、低卡")
    people: int = Field(default=1, ge=1, le=50, description="用餐人數")
    area: str = Field(default="台北市大安區", description="所在地區")


class Recommendation(BaseModel):
    """單一餐廳推薦結果 Schema（符合評分標準結構化輸出規範）。"""

    name: str = Field(..., min_length=1, description="推薦餐廳名稱")
    dishes: list[str] = Field(..., min_length=1, description="建議點購的餐點組合")
    price: int = Field(..., gt=0, description="每人預估花費金額（新台幣）")
    eta_minutes: int = Field(..., gt=0, le=120, description="預估外送送達時間（分鐘）")
    calories_estimate: int = Field(..., gt=0, description="預估餐點熱量（大卡 kcal）")
    reason: str = Field(..., min_length=10, description="符合使用者預算、口味與飲食限制的具體推薦理由")
    score: int = Field(..., ge=1, le=100, description="綜合推薦指數（1~100 分）")


class LunchDecisionResponse(BaseModel):
    """LLM 結構化輸出的頂層回應格式（包含 3 個推薦選項與工具呼叫摘要）。"""

    recommendations: list[Recommendation] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="依推薦分數排序的午餐外送推薦清單（預設 3 筆）",
    )
    weather_summary: str = Field(
        default="",
        description="結合當地天氣狀況的外送與飲食建議摘要",
    )
    budget_summary: str = Field(
        default="",
        description="多人分攤外送費後的總預算分析",
    )
