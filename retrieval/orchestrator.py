from __future__ import annotations

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from retrieval.classifier import classify_query
from retrieval.query_enhancer import enhance_query
from retrieval.hybrid_search import hybrid_search
from retrieval.answer_generator import generate_answer
from retrieval.web_agent import web_fallback_answer
from retrieval.image_interpreter import interpret_image
from pathlib import Path


# =========================
# Graph State
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]


class GraphState(TypedDict, total=False):
    user_query: str
    image_path: Optional[str]
    chat_history: List[Dict[str, str]]

    route: str
    classifier_chapter_hints: List[str]

    enhanced_queries: List[str]
    keywords: List[str]

    image_interpretation_text: str

    book_chunks: List[Dict[str, Any]]

    retrieval_distance: float
    retrieval_confidence: int

    final_answer: Dict[str, Any]
    debug: Dict[str, Any]


# =========================
# CLASSIFIER
# =========================

def classifier_node(state: GraphState):
    result = classify_query(
        state["user_query"],
        has_image=bool(state.get("image_path"))
    )

    state["route"] = result.route
    state["classifier_chapter_hints"] = result.chapter_hints or []

    state["debug"] = {
    "route": result.route,
    "classifier_reason": result.reason,
    "classifier_chapter_hints": state["classifier_chapter_hints"],
    }

    return state


# =========================
# IMAGE INTERPRETER
# =========================

def image_interpreter_node(state: GraphState):
    if not state.get("image_path"):
        return state

    interpretation = interpret_image(
        image_path=state["image_path"],
        user_query=state["user_query"]
    )

    posture_bits = []
    for p in (interpretation.posture or []):
        desc = getattr(p, "description", "")
        if desc and len(desc.split()) > 4:
            posture_bits.append(desc)

    gesture_bits = []
    for g in (interpretation.gesture_cluster or []):
        desc = getattr(g, "description", "")
        if desc and len(desc.split()) > 4:
            gesture_bits.append(desc)

    overall = getattr(interpretation, "possible_interpretation", "") or ""

    interpretation_text = " ".join(
        [
            x.strip()
            for x in (posture_bits + gesture_bits + [overall])
            if isinstance(x, str) and x.strip()
        ]
    )[:600].strip()

    state["image_interpretation_text"] = interpretation_text
    state["debug"]["image_interpretation"] = interpretation.model_dump()

    return state


# =========================
# QUERY ENHANCER
# =========================

def enhancer_node(state: GraphState):
    result = enhance_query(state["user_query"])

    state["enhanced_queries"] = result.enhanced_queries
    state["keywords"] = result.keywords

    state["debug"]["enhanced_queries"] = result.enhanced_queries
    state["debug"]["keywords"] = result.keywords

    return state


# =========================
# HYBRID SEARCH
# =========================

def hybrid_search_node(state: GraphState):
    
    search_out = hybrid_search(
        original_query=state["user_query"],
        enhanced_queries=state.get("enhanced_queries"),
        keywords=state.get("keywords"),
        image_path=state.get("image_path"),
        image_context_text=state.get("image_interpretation_text"),
        classifier_chapter_hints=state.get("classifier_chapter_hints"),
        top_k=8,
    )

    state["book_chunks"] = search_out.get("text_results", [])
    state["retrieval_distance"] = float(search_out.get("retrieval_distance") or 999.0)
    state["retrieval_confidence"] = int(search_out.get("retrieval_confidence") or 10)

    state["debug"]["retrieval_distance"] = state["retrieval_distance"]
    state["debug"]["retrieval_confidence"] = state["retrieval_confidence"]
    state["debug"]["predicted_chapters"] = search_out.get("predicted_chapters")

    return state


# =========================
# BOOK ANSWER
# =========================

def book_answer_node(state: GraphState):
    result = generate_answer(
        user_query=state["user_query"],
        book_chunks=state.get("book_chunks", []),
        retrieval_confidence=state.get("retrieval_confidence", 50),
        web_snippets=[],
        image_context_text=state.get("image_interpretation_text"),
        mode="book_only",
    )

    answerability = result.answerability_confidence
    needs_web = result.needs_web_fallback

    # If we have strong image interpretation text, trust it more
    if state.get("image_interpretation_text"):
        answerability = max(answerability, 60)
        needs_web = False

    result.answerability_confidence = answerability
    result.needs_web_fallback = needs_web

    state["final_answer"] = result.model_dump()
    state["debug"]["answerability_confidence"] = answerability
    state["debug"]["needs_web_fallback"] = needs_web
    state["debug"]["final_mode"] = "book"

    return state


# =========================
# WEB SEARCH
# =========================

def web_agent_node(state: GraphState) -> GraphState:
    q = state["user_query"]

    if state.get("image_interpretation_text"):
        q = (
            f"{q}\n\n"
            f"Image interpretation:\n"
            f"{state['image_interpretation_text']}"
        )

    result = web_fallback_answer(
        user_query=q,
        enhanced_queries=state.get("enhanced_queries"),
    )

    state["final_answer"] = result.model_dump()
    state["debug"]["final_mode"] = "web"

    return state


# =========================
# REFUSAL
# =========================

def refusal_node(state: GraphState):
    state["final_answer"] = {
        "answer": "I can only help with body language questions.",
        "retrieval_confidence": 100,
        "answerability_confidence": 100,
        "needs_web_fallback": False,
        "reasoning": "Query outside system scope.",
        "source": "System",
    }
    state["debug"]["final_mode"] = "refuse"
    return state


# =========================
# ROUTING
# =========================

def initial_route(state: GraphState):
    if state["route"] == "refuse":
        return "refuse"

    # Always interpret image first if image exists, even for web-routed queries
    if state.get("image_path"):
        return "image_interpreter"

    if state["route"] == "web":
        return "web_agent"

    return "enhancer"


def post_image_route(state: GraphState):
    # If classifier already decided web, go straight to web after image interpretation
    if state.get("route") == "web":
        return "web_agent"

    return "enhancer"


def answer_quality_route(state: GraphState):
    if state.get("debug", {}).get("needs_web_fallback", False):
        return "web"

    return "end"


# =========================
# GRAPH
# =========================

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classifier", classifier_node)
    builder.add_node("image_interpreter", image_interpreter_node)
    builder.add_node("enhancer", enhancer_node)
    builder.add_node("hybrid_search", hybrid_search_node)
    builder.add_node("book_answer", book_answer_node)
    builder.add_node("web_agent", web_agent_node)
    builder.add_node("refuse", refusal_node)

    builder.set_entry_point("classifier")

    builder.add_conditional_edges(
        "classifier",
        initial_route,
        {
            "refuse": "refuse",
            "image_interpreter": "image_interpreter",
            "web_agent": "web_agent",
            "enhancer": "enhancer",
        },
    )

    builder.add_conditional_edges(
        "image_interpreter",
        post_image_route,
        {
            "web_agent": "web_agent",
            "enhancer": "enhancer",
        },
    )

    builder.add_edge("enhancer", "hybrid_search")
    builder.add_edge("hybrid_search", "book_answer")

    builder.add_conditional_edges(
        "book_answer",
        answer_quality_route,
        {
            "web": "web_agent",
            "end": END,
        },
    )

    builder.add_edge("web_agent", END)
    builder.add_edge("refuse", END)

    return builder.compile()


# =========================
# Test
# =========================

if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(
        {
            "user_query": "Describe the body language of these people in the picture. What do their postures suggest?",
            "image_path": BASE_DIR / "test.jpg",
            "chat_history": [],
        }
    )

    print("Final Answer:", result["final_answer"])
    print("\nDebug:", result["debug"])


# Sample user queries:
# Describe the body language of these people in the picture. What do their postures suggest?
# Is this body language image on the web?
# Suggest body language tips for impressing my girlfriend.
# Latest research on microexpressions accuracy?
# What do crossed arms imply in business negotiations?
# What do dilating eyes signal?
# What is Graham's story?
# What do hands clenched together signify?
# How to tell if someone is lying to you?

# Sample image paths:
# BASE_DIR / "test.jpg"
# None
