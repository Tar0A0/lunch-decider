"""CLI 命令列介面與 Rich 視覺化輸出、結果檔案匯出模組。

執行範例：
    uv run lunch --budget 150 --taste 辣 --diet 不吃牛 --people 2
    uv run lunch --budget 150 --taste 辣 --diet 不吃牛 --people 2 --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lunch_decider.embeddings import rank_restaurants_by_embedding
from lunch_decider.llm import (
    DecisionExecutionResult,
    LLMConfig,
    decide_lunch,
    decide_lunch_mock,
    stream_chef_commentary,
)
from lunch_decider.schemas import UserQuery


def build_parser() -> argparse.ArgumentParser:
    """建立 CLI 參數解析器。"""
    parser = argparse.ArgumentParser(
        prog="lunch",
        description="🍱 今日午餐吃什麼？智慧外送決策助手 (結合 Embedding、Tool Calling、結構化輸出與串流)",
    )
    parser.add_argument("--budget", type=int, default=150, help="每人預算上限（新台幣，預設 150）")
    parser.add_argument("--taste", type=str, default="辣", help="口味偏好（例如：辣、酸辣、清爽、濃郁）")
    parser.add_argument("--diet", type=str, default="不吃牛", help="飲食限制（例如：不吃牛、全素、無）")
    parser.add_argument("--people", type=int, default=2, help="用餐人數（預設 2 人）")
    parser.add_argument("--area", type=str, default="台北市大安區", help="所在地區（預設：台北市大安區）")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "gemini"],
        default=None,
        help="指定 LLM 後端提供者（未指定時讀取環境變數 LLM_PROVIDER，預設 ollama）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="啟用離線模擬模式（不需連上真實 LLM，供本機快速體驗與 GitHub Actions CI 測試）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results",
        help="將執行結果另存為 JSON 與 Markdown 的目錄路徑（預設 docs/results）",
    )
    return parser


def save_execution_artifacts(result: DecisionExecutionResult, output_dir: Path) -> tuple[Path, Path]:
    """將執行結果另存為 `latest_decision.json` 與 `latest_decision.md` 供作品頁與 README 引用。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_decision.json"
    md_path = output_dir / "latest_decision.md"

    payload = {
        "query": result.query.model_dump(),
        "provider": result.provider,
        "model": result.model,
        "retries_used": result.retries_used,
        "embedding_candidates": result.embedding_candidates,
        "tool_calls_log": result.tool_calls_log,
        "response": result.response.model_dump(),
        "streamed_commentary": result.streamed_commentary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 🍱 今日午餐外送決策結果報告",
        "",
        "## 1. 使用者輸入條件",
        f"- **每人預算上限**：`${result.query.budget}`",
        f"- **口味偏好**：`{result.query.taste}`",
        f"- **飲食限制**：`{result.query.diet}`",
        f"- **用餐人數**：`{result.query.people} 人`",
        f"- **所在地區**：`{result.query.area}`",
        f"- **執行模型**：`{result.provider} / {result.model}`",
        "",
        "## 2. Embedding 向量相似度預篩 Top 5",
        "| 排名 | 餐廳名稱 | 料理類型 | 平均單價 | 餘弦相似度 (Cosine Similarity) |",
        "| :---: | :--- | :--- | :---: | :---: |",
    ]
    for idx, cand in enumerate(result.embedding_candidates, start=1):
        md_lines.append(
            f"| {idx} | {cand['name']} | {cand['cuisine']} | ${cand['avg_price']} | `{cand['similarity']:.4f}` |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Pydantic 結構化推薦結果 Top 3",
            "| 排名 | 推薦餐廳 | 建議餐點 | 預估單價 | 外送時間 | 熱量估算 | 推薦指數 | 推薦理由 |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for idx, rec in enumerate(result.response.recommendations, start=1):
        dishes_str = "、".join(rec.dishes)
        md_lines.append(
            f"| #{idx} | **{rec.name}** | {dishes_str} | ${rec.price} | {rec.eta_minutes} 分鐘 | {rec.calories_estimate} kcal | **{rec.score}/100** | {rec.reason} |"
        )

    md_lines.extend(
        [
            "",
            "## 4. 工具呼叫與環境分析摘要",
            f"- **天氣影響評估 (`get_weather`)**：{result.response.weather_summary}",
            f"- **預算與運費試算 (`check_budget`)**：{result.response.budget_summary}",
            "",
            "## 5. AI 外送主廚串流講評 (Streaming Output)",
            f"> {result.streamed_commentary}",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def run_cli(argv: list[str] | None = None) -> DecisionExecutionResult:
    """執行 CLI 主流程並印出 Rich 視覺化報表。"""
    # 確保 Windows 終端機能正確印出 UTF-8 與 Emoji
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console()
    parser = build_parser()
    args = parser.parse_args(argv)

    query = UserQuery(
        budget=args.budget,
        taste=args.taste,
        diet=args.diet,
        people=args.people,
        area=args.area,
    )

    console.print(
        Panel.fit(
            f"[bold yellow]💰 每人預算：[/bold yellow]${query.budget}   "
            f"[bold red]🌶️ 口味偏好：[/bold red]{query.taste}   "
            f"[bold green]🥗 飲食限制：[/bold green]{query.diet}\n"
            f"[bold cyan]👥 用餐人數：[/bold cyan]{query.people} 人     "
            f"[bold magenta]📍 所在地區：[/bold magenta]{query.area}   "
            f"[bold blue]⚙️ 執行模式：[/bold blue]{'Mock 離線驗證' if args.mock else (args.provider or '環境預設')}",
            title="[bold white on dark_orange3] 🍱 今日午餐吃什麼？智慧外送決策助手 [/bold white on dark_orange3]",
            border_style="dark_orange3",
        )
    )

    # 步驟 1：Embedding 向量相似度檢索
    embed_candidates = rank_restaurants_by_embedding(
        query=query,
        top_k=5,
        use_ollama_embed=not args.mock,
    )
    embed_table = Table(title="🧠 [能力 1：Embedding 向量語意檢索] 候選餐廳餘弦相似度 Top 5", show_lines=False)
    embed_table.add_column("排名", justify="center", style="cyan")
    embed_table.add_column("餐廳名稱", style="bold white")
    embed_table.add_column("料理類型", style="green")
    embed_table.add_column("均價", justify="right", style="yellow")
    embed_table.add_column("餘弦相似度", justify="right", style="magenta")

    for idx, item in enumerate(embed_candidates, start=1):
        embed_table.add_row(
            f"#{idx}",
            item["name"],
            item["cuisine"],
            f"${item['avg_price']}",
            f"{item['similarity']:.4f}",
        )
    console.print(embed_table)

    # 步驟 2 & 3：Tool Calling + 結構化輸出 (Pydantic 驗證)
    cfg = LLMConfig()
    if args.provider:
        cfg.provider = args.provider

    if args.mock:
        result = decide_lunch_mock(query, embedding_candidates=embed_candidates)
    else:
        try:
            result = decide_lunch(query, config=cfg, embedding_candidates=embed_candidates)
        except Exception as exc:
            console.print(
                f"[bold yellow]⚠️ 無法連線至 {cfg.provider} ({exc})，自動切換至本地決定性推理模式以完成展示。[/bold yellow]"
            )
            result = decide_lunch_mock(query, embedding_candidates=embed_candidates)

    # 顯示 Tool Calling 執行紀錄
    tool_lines = [
        f"🔧 [bold cyan]{t['tool']}[/bold cyan]({json.dumps(t['arguments'], ensure_ascii=False)})"
        for t in result.tool_calls_log
    ]
    console.print(
        Panel(
            "\n".join(tool_lines)
            + f"\n\n🌤️  [bold]天氣評估：[/bold]{result.response.weather_summary}"
            + f"\n💵 [bold]預算試算：[/bold]{result.response.budget_summary}",
            title="🛠️ [能力 2：Tool Calling 工具呼叫紀錄]",
            border_style="cyan",
        )
    )

    # 顯示 Pydantic 結構化推薦結果表格
    rec_table = Table(
        title="📋 [能力 3：Pydantic 結構化輸出] 今日外送推薦 Top 3",
        show_lines=True,
    )
    rec_table.add_column("順位", justify="center", style="bold yellow")
    rec_table.add_column("餐廳名稱与推薦餐點", style="bold white")
    rec_table.add_column("單價 / ETA / 熱量", justify="center", style="green")
    rec_table.add_column("推薦分數", justify="center", style="bold magenta")
    rec_table.add_column("AI 結構化推薦理由", style="white")

    for idx, rec in enumerate(result.response.recommendations, start=1):
        dishes_formatted = "\n".join(f"• {d}" for d in rec.dishes)
        rec_table.add_row(
            f"TOP {idx}",
            f"[bold]{rec.name}[/bold]\n[dim]{dishes_formatted}[/dim]",
            f"${rec.price}\n⏱️ {rec.eta_minutes} 分鐘\n🔥 {rec.calories_estimate} kcal",
            f"{rec.score} / 100",
            rec.reason,
        )
    console.print(rec_table)

    # 步驟 4：串流輸出主廚即時講評
    console.print("\n[bold green]⚡ [能力 4：Streaming 串流輸出] 正在接收主廚即時點餐建議：[/bold green]")
    commentary_parts: list[str] = []
    use_mock_stream = args.mock or result.provider.startswith("mock")
    for token in stream_chef_commentary(query, result.response, config=cfg, mock=use_mock_stream):
        console.print(token, end="", highlight=False)
        commentary_parts.append(token)
        if use_mock_stream and len(commentary_parts) % 8 == 0:
            time.sleep(0.005)
    console.print("\n")

    result.streamed_commentary = "".join(commentary_parts)

    # 匯出 JSON 與 Markdown 報告
    json_path, md_path = save_execution_artifacts(result, Path(args.output_dir))
    console.print(f"[dim]💾 執行結果已儲存至：{json_path.as_posix()} 與 {md_path.as_posix()}[/dim]")
    return result


def main() -> None:
    """`uv run lunch` 進入點。"""
    run_cli()


if __name__ == "__main__":
    main()
