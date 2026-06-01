"""Stage 5 — agentic investment advisor (LangGraph).

Turns the model's directional signals into actionable, budget-aware investment
plans. A small LangGraph state machine:

    load_predictions -> fetch_news -> draft_plans -> validate_allocations

reads the latest ``predictions.csv``, enriches it with market & per-stock news
from Tavily, then asks a (free-tier) LLM to draft three plans for the user's
budget:

  * aggressive   — high return / high risk
  * conservative — most-sure profit / low risk
  * optimal      — balanced, risk-adjusted

The LLM proposes per-stock rupee amounts; this module *validates and normalises*
them so every plan sums exactly to the requested budget (the model is never
trusted to do arithmetic).

Designed to degrade gracefully so the rest of QuantPilot never depends on it:
  * Tavily key / dep missing      -> plans are drafted without news.
  * LLM key / dep missing         -> a clear, surfaced error (no plans).
  * langgraph/langchain missing   -> a clear error telling the user to install.

API keys are read from the environment, or from a ``.env`` file at the project
root (parsed here with no extra dependency).
"""
from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict

import data as D
from config import CONFIG, ROOT_DIR
from logger import get_logger

log = get_logger("advisor")

# Fixed plan identities — the graph always returns these three, in this order.
PLAN_SPECS = [
    {"key": "aggressive", "title": "High Return · High Risk",
     "objective": "Maximise upside, accept volatility", "risk_level": "High"},
    {"key": "conservative", "title": "Most-Sure Profit · Low Risk",
     "objective": "Protect capital, prefer high-confidence signals", "risk_level": "Low"},
    {"key": "optimal", "title": "Optimal · Balanced",
     "objective": "Best risk-adjusted blend of the two", "risk_level": "Medium"},
]

class AdvisorState(TypedDict, total=False):
    """Shared LangGraph state. Declared in full so every channel persists across
    nodes (a bare ``dict`` schema would drop input keys not re-emitted by a node)."""
    # Inputs
    budget: float
    currency: str
    include_news: bool
    max_stocks: Optional[int]
    # Populated by nodes
    predictions: list
    as_of_date: str
    target_date: str
    market_news: list
    stock_news: dict
    news_enabled: bool
    market_outlook: str
    raw_plans: list
    plans: list
    error: str


_PROVIDER_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# ---------------------------------------------------------------------------
# Environment / config helpers
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Populate os.environ from a project-root ``.env`` (only keys not already set).

    Intentionally dependency-free: parses simple ``KEY=VALUE`` lines, ignoring
    blanks/comments and stripping optional surrounding quotes.
    """
    path = os.path.join(str(ROOT_DIR), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


def _agent_cfg() -> dict:
    return CONFIG.get("agent", {}) or {}


def _llm_cfg() -> dict:
    return _agent_cfg().get("llm", {}) or {}


def _news_cfg() -> dict:
    return _agent_cfg().get("news", {}) or {}


def _provider() -> str:
    return (_llm_cfg().get("provider") or "groq").lower()


def _llm_key_present() -> bool:
    env = _PROVIDER_ENV.get(_provider())
    return bool(env and os.environ.get(env))


def _tavily_key_present() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def _deps_installed() -> bool:
    import importlib.util as ilu
    return all(ilu.find_spec(m) for m in ("langgraph", "langchain_core"))


# ---------------------------------------------------------------------------
# LLM factory (lazy imports — keeps the web app importable without these deps)
# ---------------------------------------------------------------------------
def _make_llm():
    cfg = _llm_cfg()
    provider = _provider()
    model = cfg.get("model")
    temperature = cfg.get("temperature", 0.3)
    max_tokens = cfg.get("max_tokens", 4096)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "openai/gpt-oss-120b",
                        temperature=temperature, max_tokens=max_tokens)
    if provider in ("gemini", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash",
                                      temperature=temperature, max_output_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini",
                          temperature=temperature, max_tokens=max_tokens)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-sonnet-4-6",
                             temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown agent.llm.provider: {provider!r}")


# ---------------------------------------------------------------------------
# Tavily news retrieval
# ---------------------------------------------------------------------------
def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results,
                         topic="news", search_depth="basic")
    out = []
    for r in resp.get("results", []):
        out.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("content") or "")[:400],
            "published": r.get("published_date") or "",
        })
    return out


def _news_for_stock(name: str, ticker: str, n: int) -> tuple[str, list[dict]]:
    query = f"{name} ({ticker}) share price news latest"
    try:
        return ticker, _tavily_search(query, n)
    except Exception as exc:  # noqa: BLE001 — one stock failing must not kill the run
        log.warning("Tavily failed for %s (%s): %s", name, ticker, exc)
        return ticker, []


# ---------------------------------------------------------------------------
# JSON extraction from an LLM message
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Best-effort parse of the first JSON object in an LLM response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON object found in LLM response")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def _read_predictions() -> list[dict]:
    path = D.PREDICTIONS_PATH
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _node_load_predictions(_: dict) -> dict:
    rows = _read_predictions()
    if not rows:
        return {"error": "No predictions found. Run the predict pipeline first."}

    def _prob(r):
        try:
            return float(r.get("up_probability", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_prob, reverse=True)
    first = rows[0]
    return {
        "predictions": rows,
        "as_of_date": first.get("as_of_date", ""),
        "target_date": first.get("target_date", first.get("as_of_date", "")),
    }


def _node_fetch_news(state: dict) -> dict:
    if not state.get("include_news"):
        return {"market_news": [], "stock_news": {}, "news_enabled": False}

    ncfg = _news_cfg()
    per_stock = int(ncfg.get("per_stock_results", 2))
    market_n = int(ncfg.get("market_results", 5))
    max_stocks = state.get("max_stocks")
    if max_stocks in (None, 0):
        max_stocks = int(ncfg.get("max_stocks", 0)) or len(state["predictions"])

    # Broad market news.
    try:
        market_news = _tavily_search(
            "Indian stock market NSE Nifty 50 Sensex outlook news today", market_n)
    except Exception as exc:  # noqa: BLE001
        log.warning("Tavily market-news query failed: %s", exc)
        market_news = []

    # Per-stock news (concurrent — bounded pool so we stay polite to the API).
    stock_news: dict[str, list] = {}
    targets = state["predictions"][:max_stocks]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_news_for_stock, r.get("stock", ""), r.get("ticker", ""), per_stock)
                for r in targets]
        for fut in as_completed(futs):
            ticker, items = fut.result()
            if items:
                stock_news[ticker] = items

    log.info("News gathered: %d market items, %d stocks with news",
             len(market_news), len(stock_news))
    return {"market_news": market_news, "stock_news": stock_news, "news_enabled": True}


def _build_prompt(state: dict) -> str:
    """Render the LLM prompt, trimmed to stay within free-tier token limits.

    All predictions are included (cheap), but per-stock news is limited to the
    top ``prompt_max_stocks`` signals and each snippet is truncated — independent
    of the (larger) news set the UI displays.
    """
    ncfg = _news_cfg()
    max_news_stocks = int(ncfg.get("prompt_max_stocks", 12))
    snip = int(ncfg.get("prompt_snippet_chars", 200))

    budget = state["budget"]
    currency = state["currency"]
    preds = state["predictions"]
    lines = [
        f"INVESTMENT BUDGET: {budget:.2f} {currency}",
        f"PREDICTION TARGET DATE: {state.get('target_date')}",
        f"SIGNALS AS OF: {state.get('as_of_date')}",
        "",
        "MODEL PERFORMANCE (out-of-sample test set — the same directional model "
        "that produced the signals below):",
        "  Accuracy 0.68 | Precision 0.63 | Recall 0.74 | F1 0.68 | ROC-AUC 0.76",
        "  Per-class report (precision / recall / f1, support):",
        "    UP   (1): 0.63 / 0.74 / 0.68  (n=2444)",
        "    DOWN (0): 0.74 / 0.63 / 0.68  (n=2834)",
        "  Read: the model catches most real UP moves (UP recall ~0.74) but UP "
        "precision is only ~0.63, so expect a fair share of false UP calls. "
        "Treat up_probability as useful-but-imperfect confidence, not a guarantee, "
        "and size risk accordingly.",
        "",
        "MODEL SIGNALS (sorted by up-probability; higher = more confident UP):",
    ]
    for r in preds:
        lines.append(
            f"- {r.get('stock')} [{r.get('ticker')}]: signal={r.get('signal')}, "
            f"up_probability={r.get('up_probability')}"
        )

    if state.get("market_news"):
        lines += ["", "BROAD INDIAN-MARKET NEWS:"]
        for n in state["market_news"]:
            lines.append(f"- {n['title']} — {n['content'][:snip]}")

    stock_news = state.get("stock_news") or {}
    if stock_news:
        # Only feed news for the highest-confidence signals (preds is sorted).
        top = preds if not max_news_stocks else preds[:max_news_stocks]
        lines += ["", "PER-STOCK NEWS (top signals):"]
        for r in top:
            items = stock_news.get(r.get("ticker"))
            if not items:
                continue
            for n in items:
                lines.append(f"- [{r.get('ticker')}] {n['title']} — {n['content'][:snip]}")

    return "\n".join(lines)


_SYSTEM = """You are QuantPilot's portfolio strategist for Indian (NSE) equities.
You are given an ML model's next-day directional signals (with up-probabilities),
recent news, and a fixed investment budget. Draft EXACTLY three allocation plans.

Each plan has a STRICT eligibility threshold on up_probability — never allocate
to a stock whose up_probability is at or below the plan's cutoff.

Plans (use these exact keys):
- "aggressive": high return / high risk. ELIGIBLE: up_probability > 0.5.
  Within those, concentrate in the highest-probability names; tolerate
  volatility. Few names, larger bets.
- "conservative": most-sure profit / low risk. ELIGIBLE: up_probability > 0.6
  (only the strongest, news-corroborated UP signals). Diversify across them, and
  you MAY hold part as uninvested cash (ticker "CASH", stock "Cash (uninvested)")
  to reduce risk — especially if few names clear 0.6.
- "optimal": balanced, best risk-adjusted blend. ELIGIBLE: up_probability > 0.4.
  Weight allocations toward higher-probability, news-supported names.

Rules:
- Apply each plan's up_probability threshold strictly; if a name doesn't clear a
  plan's cutoff, it cannot appear in that plan.
- Allocate the WHOLE budget (amounts are absolute, in the budget currency); each
  plan's amounts should sum to roughly the full budget. If too few names qualify,
  size up the qualifying names (aggressive/optimal) or hold the remainder as CASH
  (conservative). Higher up_probability should generally get a larger share.
- Ground each pick in the signal AND any relevant news; keep reasons short.
- Output ONLY a single JSON object, no prose, no markdown fences.

JSON shape:
{
  "market_outlook": "2-3 sentence read on the broad market",
  "plans": [
    {
      "key": "aggressive",
      "summary": "1-2 sentences on the strategy",
      "expected_return": "qualitative or rough % range",
      "allocations": [
        {"stock": "Reliance Industries", "ticker": "RELIANCE.NS",
         "amount": 25000, "reason": "short reason"}
      ]
    }
    // ...conservative, optimal
  ]
}"""


def _node_draft_plans(state: dict) -> dict:
    try:
        llm = _make_llm()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"LLM unavailable: {exc}"}

    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = _build_prompt(state)
    try:
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = _extract_json(text)
    except Exception as exc:  # noqa: BLE001 — surface a clean message to the UI
        return {"error": f"Plan generation failed: {exc}"}

    return {
        "market_outlook": parsed.get("market_outlook", ""),
        "raw_plans": parsed.get("plans", []),
    }


def _normalise_allocations(allocs: list[dict], budget: float) -> tuple[list[dict], float]:
    """Coerce, drop junk, and scale allocations so they sum exactly to ``budget``."""
    clean = []
    for a in allocs or []:
        try:
            amt = float(a.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt <= 0:
            continue
        clean.append({
            "stock": str(a.get("stock", "")).strip() or "—",
            "ticker": str(a.get("ticker", "")).strip() or "—",
            "amount": amt,
            "reason": str(a.get("reason", "")).strip(),
        })

    total = sum(a["amount"] for a in clean)
    if not clean or total <= 0:
        return [], 0.0

    scale = budget / total
    for a in clean:
        a["amount"] = round(a["amount"] * scale, 2)
    # Absorb rounding drift into the largest position so the sum is exact.
    drift = round(budget - sum(a["amount"] for a in clean), 2)
    if abs(drift) >= 0.01:
        biggest = max(clean, key=lambda x: x["amount"])
        biggest["amount"] = round(biggest["amount"] + drift, 2)

    for a in clean:
        a["percent"] = round(a["amount"] / budget * 100, 2)
    clean.sort(key=lambda x: x["amount"], reverse=True)
    return clean, round(sum(a["amount"] for a in clean), 2)


def _node_validate(state: dict) -> dict:
    budget = state["budget"]
    raw_by_key = {p.get("key"): p for p in state.get("raw_plans", []) if isinstance(p, dict)}

    plans = []
    for spec in PLAN_SPECS:
        raw = raw_by_key.get(spec["key"], {})
        allocs, total = _normalise_allocations(raw.get("allocations", []), budget)
        plans.append({
            "key": spec["key"],
            "title": spec["title"],
            "objective": spec["objective"],
            "risk_level": spec["risk_level"],
            "summary": str(raw.get("summary", "")).strip(),
            "expected_return": str(raw.get("expected_return", "")).strip(),
            "allocations": allocs,
            "total": total,
        })
    return {"plans": plans}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def _build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(AdvisorState)
    g.add_node("load", _node_load_predictions)
    g.add_node("news", _node_fetch_news)
    g.add_node("draft", _node_draft_plans)
    g.add_node("validate", _node_validate)

    g.set_entry_point("load")
    g.add_conditional_edges(
        "load", lambda s: "stop" if s.get("error") else "go",
        {"stop": END, "go": "news"})
    g.add_edge("news", "draft")
    g.add_conditional_edges(
        "draft", lambda s: "stop" if s.get("error") else "go",
        {"stop": END, "go": "validate"})
    g.add_edge("validate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def advisor_status() -> dict:
    """Report readiness so the UI can guide the user (never raises)."""
    _load_dotenv()
    provider = _provider()
    deps = _deps_installed()
    return {
        "deps_installed": deps,
        "llm": {
            "provider": provider,
            "model": _llm_cfg().get("model"),
            "key_env": _PROVIDER_ENV.get(provider),
            "key_present": _llm_key_present(),
        },
        "tavily": {"key_present": _tavily_key_present()},
        "news_enabled": bool(_news_cfg().get("enabled", True)),
        "currency": _agent_cfg().get("currency", "INR"),
        "ready": deps and _llm_key_present(),
    }


def generate_plans(budget: float,
                   currency: Optional[str] = None,
                   include_news: Optional[bool] = None,
                   max_stocks: Optional[int] = None) -> dict[str, Any]:
    """Run the LangGraph advisor and return three budget-aware plans.

    Returns ``{"ok": True, ...}`` on success or ``{"ok": False, "error": ...}``
    on any handled failure (missing deps/keys, no predictions, LLM error).
    """
    _load_dotenv()

    if budget is None or budget <= 0:
        return {"ok": False, "error": "Budget must be a positive number."}
    if not _deps_installed():
        return {"ok": False, "error": (
            "Advisor dependencies are not installed. Run: "
            "pip install langgraph langchain-core langchain-groq tavily-python")}
    if not _llm_key_present():
        env = _PROVIDER_ENV.get(_provider())
        return {"ok": False, "error": (
            f"LLM API key not found. Set {env} in the environment or a .env file "
            f"at the project root (provider='{_provider()}').")}

    news_enabled = (
        bool(_news_cfg().get("enabled", True)) if include_news is None else bool(include_news)
    )
    if news_enabled and not _tavily_key_present():
        log.info("TAVILY_API_KEY missing — drafting plans without news.")
        news_enabled = False

    currency = currency or _agent_cfg().get("currency", "INR")

    initial = {
        "budget": float(budget),
        "currency": currency,
        "include_news": news_enabled,
        "max_stocks": max_stocks,
    }

    try:
        graph = _build_graph()
        final = graph.invoke(initial)
    except Exception as exc:  # noqa: BLE001
        log.exception("Advisor graph failed")
        return {"ok": False, "error": f"Advisor failed: {exc}"}

    if final.get("error"):
        return {"ok": False, "error": final["error"]}

    return {
        "ok": True,
        "budget": float(budget),
        "currency": currency,
        "as_of_date": final.get("as_of_date", ""),
        "target_date": final.get("target_date", ""),
        "market_outlook": final.get("market_outlook", ""),
        "plans": final.get("plans", []),
        "market_news": final.get("market_news", []),
        "stock_news": final.get("stock_news", {}),
        "news_enabled": final.get("news_enabled", False),
        "generated_with": {
            "provider": _provider(),
            "model": _llm_cfg().get("model"),
            "news": final.get("news_enabled", False),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantPilot agentic investment advisor")
    parser.add_argument("--budget", type=float, required=True, help="Amount to invest")
    parser.add_argument("--no-news", action="store_true", help="Skip Tavily news")
    args = parser.parse_args()

    result = generate_plans(args.budget, include_news=not args.no_news)
    print(json.dumps(result, indent=2, ensure_ascii=False))
