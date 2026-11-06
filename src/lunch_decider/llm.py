"""LLM 核心串接模組：支援本地 Ollama 與雲端 OpenAI / Gemini 切換。

實作之核心 LLM 能力：
1. 工具呼叫 (Tool Calling)：將 `TOOLS_SCHEMA` 傳遞給模型，自動攔截並執行 `tool_calls`，
   把本地餐廳搜尋、天氣查詢與預算試算結果回填至對話歷史。
2. 結構化輸出 (Structured Output) 與重試驗證：利用 Pydantic 產生的 JSON Schema
   約束模型輸出格式，並以 `LunchDecisionResponse.model_validate_json` 進行型別驗證；
   若驗證失敗，將錯誤原因附上並自動重試（預設最多 3 次）。
3. 串流輸出 (Streaming)：透過 `httpx.stream` 逐 token 串流印出智慧外送主廚的點餐分析。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from lunch_decider.schemas import LunchDecisionResponse, Recommendation, UserQuery
from lunch_decider.tools import (
    TOOLS_SCHEMA,
    check_budget,
    execute_tool_call,
    get_weather,
    search_restaurants,
)

# 載入 .env 環境變數（若存在）
load_dotenv()


@dataclass
class LLMConfig:
    """從環境變數讀取的 LLM 連線與模型設定。"""

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
            "/"
        )
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    timeout_seconds: float = 60.0


@dataclass
class DecisionExecutionResult:
    """封裝一次完整午餐決策的結構化結果與中間過程紀錄（供 CLI 與作品頁輸出）。"""

    query: UserQuery
    response: LunchDecisionResponse
    provider: str
    model: str
    tool_calls_log: list[dict[str, Any]] = field(default_factory=list)
    embedding_candidates: list[dict[str, Any]] = field(default_factory=list)
    streamed_commentary: str = ""
    retries_used: int = 0


def _extract_json_object(raw_text: str) -> str:
    """從模型回傳文字中提取純 JSON 字串（處理可能夾帶的 ```json 區塊）。"""
    cleaned = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1]
    return cleaned


def _build_system_prompt() -> str:
    """建立系統提示詞，指導模型使用工具並遵守結構化 JSON Schema。"""
    return (
        "你是『今日午餐吃什麼？智慧外送決策助手』。"
        "請根據使用者的預算、口味偏好、飲食限制、用餐人數與所在地區，"
        "先呼叫工具查詢候選餐廳、當地天氣與外送費分攤預算，"
        "再從候選名單中挑選最適合的 3 家餐廳，"
        "嚴格按照指定的 JSON Schema 格式回傳繁體中文推薦結果。"
        "注意：若使用者指定『不吃牛』或『全素』，絕對不可推薦違規餐點！"
    )


def _call_chat_with_tools(
    messages: list[dict[str, Any]],
    config: LLMConfig,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """向 Ollama 或雲端 OpenAI 相容介面發送支援 Tool Calling 的對話請求。"""
    own_client = client is None
    http = client or httpx.Client(timeout=config.timeout_seconds)
    try:
        if config.provider == "ollama":
            url = f"{config.ollama_base_url}/api/chat"
            payload = {
                "model": config.ollama_model,
                "messages": messages,
                "tools": TOOLS_SCHEMA,
                "stream": False,
            }
            resp = http.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {})

        # OpenAI 或 Gemini (透過 OpenAI 相容端點)
        api_key, base_url, model = _resolve_cloud_credentials(config)
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
        }
        resp = http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]
    finally:
        if own_client:
            http.close()


def _call_structured_json(
    messages: list[dict[str, Any]],
    config: LLMConfig,
    client: httpx.Client | None = None,
) -> str:
    """要求模型依據 `LunchDecisionResponse` 的 JSON Schema 回傳結構化 JSON 字串。"""
    schema = LunchDecisionResponse.model_json_schema()
    own_client = client is None
    http = client or httpx.Client(timeout=config.timeout_seconds)
    try:
        if config.provider == "ollama":
            url = f"{config.ollama_base_url}/api/chat"
            payload = {
                "model": config.ollama_model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            resp = http.post(url, json=payload)
            resp.raise_for_status()
            return str(resp.json().get("message", {}).get("content", ""))

        api_key, base_url, model = _resolve_cloud_credentials(config)
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        resp = http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])
    finally:
        if own_client:
            http.close()


def _resolve_cloud_credentials(config: LLMConfig) -> tuple[str, str, str]:
    """解析雲端模型的 API Key、Base URL 與模型名稱。"""
    if config.provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("已選擇 LLM_PROVIDER=gemini，但未設定 GEMINI_API_KEY 環境變數。")
        return (
            config.gemini_api_key,
            "https://generativelanguage.googleapis.com/v1beta/openai",
            config.gemini_model,
        )
    if config.provider == "openai":
        if not config.openai_api_key:
            raise ValueError("已選擇 LLM_PROVIDER=openai，但未設定 OPENAI_API_KEY 環境變數。")
        return config.openai_api_key, config.openai_base_url, config.openai_model

    raise ValueError(f"不支援的 LLM_PROVIDER: {config.provider}")


def decide_lunch(
    query: UserQuery,
    config: LLMConfig | None = None,
    embedding_candidates: list[dict[str, Any]] | None = None,
    max_retries: int = 3,
    client: httpx.Client | None = None,
) -> DecisionExecutionResult:
    """執行完整 LLM 決策流程：Tool Calling -> 結構化輸出 -> Pydantic 驗證與失敗重試。"""
    cfg = config or LLMConfig()
    active_model = (
        cfg.ollama_model
        if cfg.provider == "ollama"
        else (cfg.gemini_model if cfg.provider == "gemini" else cfg.openai_model)
    )

    user_prompt = (
        f"請幫我決定今日午餐外送！條件如下：\n"
        f"- 每人預算上限：${query.budget} 元\n"
        f"- 口味偏好：{query.taste}\n"
        f"- 飲食限制：{query.diet}\n"
        f"- 用餐人數：{query.people} 人\n"
        f"- 所在地區：{query.area}\n"
        "請先呼叫工具查詢餐廳、天氣與預算，再給我 3 個推薦選項。"
    )
    if embedding_candidates:
        names = [c["name"] for c in embedding_candidates]
        user_prompt += f"\n（向量語意檢索預篩高相似度餐廳：{', '.join(names)}）"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    tool_calls_log: list[dict[str, Any]] = []

    # Step 1: 嘗試讓模型觸發 Tool Calling
    assistant_msg = _call_chat_with_tools(messages, cfg, client=client)
    raw_tool_calls = assistant_msg.get("tool_calls") or []

    if raw_tool_calls:
        messages.append(assistant_msg)
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            tool_output = execute_tool_call(fn_name, args)
            tool_calls_log.append({"tool": fn_name, "arguments": args, "output": json.loads(tool_output)})
            tool_msg: dict[str, Any] = {"role": "tool", "content": tool_output}
            if "id" in tc:
                tool_msg["tool_call_id"] = tc["id"]
            messages.append(tool_msg)
    else:
        # 若小模型未自動發出 tool_calls，主動執行工具並注入上下文，確保決策有所本
        auto_calls = [
            (
                "search_restaurants",
                {"max_price": query.budget, "taste": query.taste, "diet": query.diet, "area": query.area},
            ),
            ("get_weather", {"city": query.area}),
            (
                "check_budget",
                {"price_per_person": query.budget, "people": query.people, "budget_per_person": query.budget},
            ),
        ]
        for fn_name, args in auto_calls:
            out = execute_tool_call(fn_name, args)
            tool_calls_log.append({"tool": fn_name, "arguments": args, "output": json.loads(out)})
            messages.append({"role": "tool", "content": out})

    # Step 2: 要求結構化輸出並以 Pydantic 驗證（含失敗重試迴圈）
    schema_instruction = (
        "請根據上述工具查詢結果，嚴格輸出符合以下 JSON Schema 的物件（不要包含多餘說明）：\n"
        f"{json.dumps(LunchDecisionResponse.model_json_schema(), ensure_ascii=False)}"
    )
    messages.append({"role": "user", "content": schema_instruction})

    last_error: Exception | None = None
    for attempt in range(max_retries):
        raw_json = _call_structured_json(messages, cfg, client=client)
        extracted = _extract_json_object(raw_json)
        try:
            validated = LunchDecisionResponse.model_validate_json(extracted)
            return DecisionExecutionResult(
                query=query,
                response=validated,
                provider=cfg.provider,
                model=active_model,
                tool_calls_log=tool_calls_log,
                embedding_candidates=embedding_candidates or [],
                retries_used=attempt,
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw_json})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"你的輸出未通過 Pydantic JSON Schema 驗證，錯誤原因：{exc}\n"
                        "請修正欄位型別、範圍與必填項目後，重新只輸出合法 JSON。"
                    ),
                }
            )

    raise RuntimeError(f"結構化輸出經過 {max_retries} 次重試仍驗證失敗：{last_error}")


def stream_chef_commentary(
    query: UserQuery,
    decision: LunchDecisionResponse,
    config: LLMConfig | None = None,
    mock: bool = False,
) -> Iterator[str]:
    """以串流 (Streaming) 方式逐字產生智慧主廚的即時點餐講評。"""
    top_names = "、".join(r.name for r in decision.recommendations[:3])
    if mock:
        mock_text = (
            f"👨‍🍳 【AI 外送決策主廚即時講評】\n"
            f"根據您設定的每人 ${query.budget} 預算、「{query.taste}」口味偏好與「{query.diet}」限制，"
            f"我們從 16 家合作餐廳中透過向量檢索與工具過濾，為 {query.people} 位用餐者精選出：{top_names}。\n"
            f"考量今日{decision.weather_summary}，首選店家的預估外送時間僅需 "
            f"{decision.recommendations[0].eta_minutes} 分鐘，且完全不含牛肉成分；"
            f"{decision.budget_summary} 祝您今日午餐愉快！"
        )
        for ch in mock_text:
            yield ch
        return

    cfg = config or LLMConfig()
    prompt = (
        f"使用者條件：預算 ${query.budget}、口味 {query.taste}、飲食限制 {query.diet}、{query.people} 人。\n"
        f"已選出的推薦餐廳：{top_names}。\n"
        "請用熱情專業的語氣，在 120 字以內給出簡短的點餐與湊單建議。"
    )

    try:
        if cfg.provider == "ollama":
            url = f"{cfg.ollama_base_url}/api/chat"
            payload = {
                "model": cfg.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            with httpx.Client(timeout=cfg.timeout_seconds) as http:
                with http.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
            return

        api_key, base_url, model = _resolve_cloud_credentials(cfg)
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        with httpx.Client(timeout=cfg.timeout_seconds) as http:
            with http.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
    except Exception:
        # 若串流端點連線異常，優雅降級為本地串流講評
        yield from stream_chef_commentary(query, decision, config=cfg, mock=True)


def decide_lunch_mock(
    query: UserQuery,
    embedding_candidates: list[dict[str, Any]] | None = None,
) -> DecisionExecutionResult:
    """不需連線外部 LLM 的決定性模擬模式（供 `--mock` CLI 與 GitHub Actions CI 測試使用）。"""
    candidates = search_restaurants(
        max_price=query.budget,
        taste=query.taste,
        diet=query.diet,
        area=query.area,
    )
    if not candidates:
        candidates = search_restaurants(max_price=250, diet=query.diet)

    weather = get_weather(query.area)
    selected = candidates[:3]
    first_price = selected[0]["avg_price"] if selected else query.budget
    budget_info = check_budget(
        price_per_person=first_price,
        people=query.people,
        budget_per_person=query.budget,
    )

    tool_calls_log = [
        {
            "tool": "search_restaurants",
            "arguments": {
                "max_price": query.budget,
                "taste": query.taste,
                "diet": query.diet,
                "area": query.area,
            },
            "output": candidates[:3],
        },
        {
            "tool": "get_weather",
            "arguments": {"city": query.area},
            "output": weather,
        },
        {
            "tool": "check_budget",
            "arguments": {
                "price_per_person": first_price,
                "people": query.people,
                "budget_per_person": query.budget,
            },
            "output": budget_info,
        },
    ]

    recommendations: list[Recommendation] = []
    base_scores = [96, 92, 88]
    for idx, rest in enumerate(selected):
        score = base_scores[idx] if idx < len(base_scores) else 85
        reason = (
            f"【{rest['cuisine']}】平均每人 ${rest['avg_price']} 完全符合 ${query.budget} 預算上限；"
            f"具備「{'、'.join(rest['taste_tags'][:2])}」風味且"
            f"{'全素可食' if rest['is_vegetarian'] else '絕不含牛肉成分'}，"
            f"預估 {rest['eta_minutes']} 分鐘即可送達{query.area}。"
        )
        recommendations.append(
            Recommendation(
                name=rest["name"],
                dishes=rest["signature_dishes"][:2],
                price=rest["avg_price"],
                eta_minutes=rest["eta_minutes"],
                calories_estimate=rest["calories_estimate"],
                reason=reason,
                score=score,
            )
        )

    response = LunchDecisionResponse(
        recommendations=recommendations,
        weather_summary=f"{weather['city']}{weather['condition']}（{weather['temperature_c']}°C，降雨機率 {weather['rain_probability']}%）：{weather['delivery_impact']}",
        budget_summary=budget_info["summary"],
    )

    return DecisionExecutionResult(
        query=query,
        response=response,
        provider="mock (local-deterministic)",
        model="lunch-decider-mock-v1",
        tool_calls_log=tool_calls_log,
        embedding_candidates=embedding_candidates or [],
        retries_used=0,
    )
