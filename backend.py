from __future__ import annotations

import operator
import os
import re
import json
import base64
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, Union, Dict, Any

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int = Field(default=1)
    title: str = Field(default="Section Title")
    goal: str = Field(default="Understand key details", description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(default_factory=list)
    target_words: int = Field(default=300, description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str = "Technical Blog Outline"
    audience: str = "Software Engineers & Developers"
    tone: str = "Informative and technical"
    blog_kind: str = "explainer"
    constraints: Union[List[Any], Dict[str, Any], Any] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str = ""
    url: str = ""
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool = True
    mode: Literal["closed_book", "hybrid", "open_book"] = "hybrid"
    reason: Optional[str] = ""
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (removed) ----

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer
    merged_md: str
    final: str


# -----------------------------
# 2) LLM
# -----------------------------
# Groq 1 for structured output
groq_llm_1 = ChatGroq(
    model=os.getenv("GROQ_MODEL_1", "openai/gpt-oss-20b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=5,
)

# Groq 2 for worker text generation
groq_llm_2 = ChatGroq(
    model=os.getenv("GROQ_MODEL_2", "openai/gpt-oss-20b"),
    api_key=os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=5,
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.

Return your response in valid JSON with keys: "needs_research" (boolean), "mode" ("closed_book"|"hybrid"|"open_book"), "reason" (string), and "queries" (list of strings).
"""

def _parse_json(text: str) -> dict:
    """Extract the first JSON object from LLM text output."""
    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def router_node(state: State) -> dict:
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ]
    raw = groq_llm_1.invoke(messages).content
    data = _parse_json(raw)
    decision = RouterDecision(**data)

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            snippet = r.get("content") or r.get("snippet") or ""
            # Limit snippet length to avoid TPM limit errors
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": snippet,
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.

Return your response in valid JSON formatted as a JSON object with key "evidence" containing a list of objects with fields "title", "url", "published_at", "snippet", "source".
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:3]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=2))

    raw = raw[:6]
    if not raw:
        return {"evidence": []}

    # Skip LLM extraction — just convert raw Tavily results directly
    evidence_items = [
        EvidenceItem(
            title=r.get("title") or "",
            url=r.get("url") or "",
            snippet=(r.get("snippet") or "")[:200],
            source=r.get("source") or "",
            published_at=r.get("published_at"),
        )
        for r in raw
    ]

    dedup = {}
    for e in evidence_items:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks (sections). Each task MUST include:
  * "id": integer starting from 1 (1, 2, 3, ...)
  * "title": a specific, descriptive, and engaging section title (e.g. "Overview of RAG Architecture", "Top 2026 Framework Comparison", "Enterprise Use Cases", etc.). NEVER use generic titles like "Section Title" or "Section".
  * "goal": one sentence goal describing what the reader should understand.
  * "bullets": 3–6 specific bullet points to cover in this section.
  * "target_words": integer word count (120–550).
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Return your output as a valid JSON object matching the Plan schema with keys: "blog_title", "audience", "tone", "blog_kind", "constraints", and "tasks".
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    messages = [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n"
                f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                f"Evidence:\n{[e.model_dump() for e in evidence][:8]}"
            )
        ),
    ]

    raw = groq_llm_1.invoke(messages).content
    plan = _parse_json(raw)
    if isinstance(plan, dict):
        constraints_raw = plan.get("constraints")
        if isinstance(constraints_raw, dict):
            plan["constraints"] = [f"{k}: {v}" for k, v in constraints_raw.items()]
        elif isinstance(constraints_raw, str):
            plan["constraints"] = [constraints_raw]

        tasks_raw = plan.get("tasks", [])
        for idx, t in enumerate(tasks_raw):
            if isinstance(t, dict):
                t["id"] = idx + 1
                title_val = t.get("title") or t.get("name") or t.get("section_title") or t.get("header") or ""
                if not title_val or title_val.strip().lower() in ["section title", "section", "untitled", "task title"]:
                    goal_text = t.get("goal", "")
                    bullets = t.get("bullets", [])
                    if goal_text:
                        clean_goal = goal_text.split(".")[0].strip()
                        title_val = clean_goal[:60]
                    elif bullets:
                        title_val = bullets[0][:60]
                    else:
                        title_val = f"Key Takeaways Part {idx + 1}"
                t["title"] = title_val
        plan = Plan(**plan)
    elif hasattr(plan, "tasks"):
        if isinstance(getattr(plan, "constraints", None), dict):
            plan.constraints = [f"{k}: {v}" for k, v in plan.constraints.items()]
        for idx, task in enumerate(plan.tasks):
            task.id = idx + 1
            if not task.title or task.title.strip().lower() in ["section title", "section", "untitled", "task title"]:
                if task.goal:
                    clean_goal = task.goal.split(".")[0].strip()
                    task.title = clean_goal[:60]
                elif task.bullets:
                    task.title = task.bullets[0][:60]
                else:
                    task.title = f"Key Takeaways Part {idx + 1}"
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output must start with the section title header: "## <Section Title>" using the exact Section Title provided in the prompt (e.g. ## Overview of Modern RAG Architecture).

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = groq_llm_2.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) Reducer (subgraph)
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md, "final": merged_md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
