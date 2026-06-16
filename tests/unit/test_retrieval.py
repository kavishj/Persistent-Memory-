"""
tests/unit/test_retrieval.py

Day 6 unit tests for the retrieval engine.
Tests reranker and context assembler with mocked memory data.
Does NOT require live Weaviate or Postgres — all data is in-memory.

Run:
    python -m pytest tests/unit/test_retrieval.py -v
"""

from datetime import datetime, timezone, timedelta
from core.retrieval.query_builder import RawMemory, QueryResult
from core.retrieval.reranker import rerank, _final_score
from core.retrieval.context_assembler import (
    assemble_context,
    BUDGET_HARD_CEILING,
    SLOT_PROCEDURAL,
    SLOT_SEMANTIC_NO_PROC,
    _estimate_tokens,
)


# ---------------------------------------------------------------------------
# Helpers — build test memories
# ---------------------------------------------------------------------------
def make_semantic(
    postgres_id: str,
    fact: str,
    retrieval_score: float = 0.8,
    importance_score: float = 0.7,
    confidence: float = 0.85,
    days_old: int = 3,
    access_count: int = 5,
) -> RawMemory:
    last_confirmed = datetime.now(timezone.utc) - timedelta(days=days_old)
    return RawMemory(
        postgres_id=postgres_id,
        memory_type="semantic",
        content=fact,
        retrieval_score=retrieval_score,
        importance_score=importance_score,
        confidence=confidence,
        last_confirmed=last_confirmed,
        properties={
            "fact": fact,
            "fact_type": "constraint",
            "confidence": confidence,
            "importance_score": importance_score,
            "access_count": access_count,
            "last_confirmed": last_confirmed.isoformat(),
            "postgres_id": postgres_id,
            "entities": ["payments API"],
            "scope": "agent",
        },
    )


def make_procedural(
    postgres_id: str,
    trigger: str,
    task_type: str = "etl_run",
    confidence: float = 0.88,
    importance_score: float = 0.85,
) -> RawMemory:
    return RawMemory(
        postgres_id=postgres_id,
        memory_type="procedural",
        content=trigger,
        retrieval_score=0.9,
        importance_score=importance_score,
        confidence=confidence,
        properties={
            "trigger_condition": trigger,
            "task_type": task_type,
            "confidence": confidence,
            "importance_score": importance_score,
            "postgres_id": postgres_id,
            "detail": {
                "steps": [
                    {
                        "step_num": 1,
                        "action": "Check source table row count",
                        "rationale": "Determines batch size",
                        "tool_hint": "db_query_tool",
                    },
                    {
                        "step_num": 2,
                        "action": "Set batch size to min(row_count/1000, 1000)",
                        "rationale": "DB drops connections above 5000 rows",
                        "tool_hint": None,
                    },
                    {
                        "step_num": 3,
                        "action": "Run ETL in batches with 500ms sleep",
                        "rationale": "Prevents connection pool exhaustion",
                        "tool_hint": "etl_runner_tool",
                    },
                ],
                "edge_cases": [
                    {
                        "condition": "connection timeout occurs mid-run",
                        "modification": "Resume from last committed batch_id",
                    }
                ],
                "expected_outcome": "All rows transferred, counts match",
            },
        },
    )


def make_episodic(
    postgres_id: str,
    task_prompt: str,
    outcome: str = "success",
    days_old: int = 5,
    importance_score: float = 0.6,
) -> RawMemory:
    session_start = datetime.now(timezone.utc) - timedelta(days=days_old)
    return RawMemory(
        postgres_id=postgres_id,
        memory_type="episodic",
        content=task_prompt,
        retrieval_score=0.7,
        importance_score=importance_score,
        properties={
            "task_prompt": task_prompt,
            "task_type": "etl_run",
            "outcome": outcome,
            "session_start": session_start.isoformat(),
            "importance_score": importance_score,
            "access_count": 2,
            "last_confirmed": session_start.isoformat(),
            "postgres_id": postgres_id,
        },
    )


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------
def test_reranker_sorts_semantic_by_final_score():
    """Higher retrieval+importance+recency should rank first."""
    high = make_semantic("id-1", "High quality fact", retrieval_score=0.9,
                         importance_score=0.9, days_old=1)
    low  = make_semantic("id-2", "Low quality fact",  retrieval_score=0.3,
                         importance_score=0.2, days_old=60)

    result = QueryResult(semantic=[low, high], procedural=None, episodic=[])
    reranked = rerank(result)

    assert reranked.semantic[0].postgres_id == "id-1", (
        f"Expected id-1 first, got {reranked.semantic[0].postgres_id}"
    )


def test_reranker_trims_semantic_to_k5():
    """Reranker must return at most 5 semantic memories."""
    mems = [
        make_semantic(f"id-{i}", f"Fact {i}", retrieval_score=0.8 - i * 0.05)
        for i in range(10)
    ]
    result = QueryResult(semantic=mems, procedural=None, episodic=[])
    reranked = rerank(result)

    assert len(reranked.semantic) <= 5, (
        f"Expected ≤5 semantic results, got {len(reranked.semantic)}"
    )


def test_reranker_trims_episodic_to_k3():
    """Reranker must return at most 3 episodic memories."""
    mems = [
        make_episodic(f"ep-{i}", f"Task prompt {i}")
        for i in range(8)
    ]
    result = QueryResult(semantic=[], procedural=None, episodic=mems)
    reranked = rerank(result)

    assert len(reranked.episodic) <= 3, (
        f"Expected ≤3 episodic results, got {len(reranked.episodic)}"
    )


def test_reranker_passes_procedural_unchanged():
    """Procedural memory must pass through reranker without modification."""
    proc = make_procedural("proc-1", "Running nightly ETL for large tables")
    result = QueryResult(semantic=[], procedural=proc, episodic=[])
    reranked = rerank(result)

    assert reranked.procedural is proc, "Procedural should be the same object"
    assert reranked.procedural.postgres_id == "proc-1"


def test_reranker_deduplicates_semantic():
    """Same postgres_id from agent + global tenant should appear only once."""
    mem_a = make_semantic("dup-id", "Payments API rate limit is 100/min",
                          retrieval_score=0.9)
    mem_b = make_semantic("dup-id", "Payments API rate limit is 100/min",
                          retrieval_score=0.7)

    result = QueryResult(semantic=[mem_a, mem_b], procedural=None, episodic=[])
    reranked = rerank(result)

    ids = [m.postgres_id for m in reranked.semantic]
    assert ids.count("dup-id") == 1, (
        f"Duplicate postgres_id should appear once, got {ids.count('dup-id')}"
    )


def test_reranker_stale_memory_ranks_lower():
    """60-day-old memory should rank below fresh memory of same retrieval score."""
    fresh = make_semantic("fresh", "Fresh fact", retrieval_score=0.75,
                          importance_score=0.7, days_old=1)
    stale = make_semantic("stale", "Stale fact", retrieval_score=0.75,
                          importance_score=0.7, days_old=60, access_count=0)

    result = QueryResult(semantic=[stale, fresh], procedural=None, episodic=[])
    reranked = rerank(result)

    assert reranked.semantic[0].postgres_id == "fresh", (
        "Fresh memory should rank above stale memory with same retrieval score"
    )


# ---------------------------------------------------------------------------
# Context assembler tests
# ---------------------------------------------------------------------------
def test_assembler_includes_procedural_when_found():
    """Procedural memory must appear in context string when found."""
    proc = make_procedural("proc-1", "Running nightly ETL for large tables")
    sem  = [make_semantic("sem-1", "DB drops connections above 5000 rows")]
    result = QueryResult(semantic=sem, procedural=proc, episodic=[])

    ctx = assemble_context(result)

    assert ctx.procedural_found is True
    assert "Procedure" in ctx.context_string
    assert "etl_run" in ctx.context_string


def test_assembler_no_procedural_expands_semantic_slot():
    """Without procedural, semantic slot should expand to 1000 tokens."""
    # Create enough semantic memories to fill the expanded slot
    mems = [
        make_semantic(f"sem-{i}", f"Fact number {i}: " + "x" * 100)
        for i in range(20)
    ]
    result = QueryResult(semantic=mems, procedural=None, episodic=[])
    ctx = assemble_context(result)

    assert ctx.procedural_found is False
    # Semantic slot is 1000 tokens without procedural — should fit more facts
    assert ctx.semantic_count >= 3, (
        f"Expected ≥3 semantic facts with expanded slot, got {ctx.semantic_count}"
    )


def test_assembler_respects_hard_ceiling():
    """Context string must never exceed BUDGET_HARD_CEILING tokens."""
    # Create very large memories
    big_fact = "A" * 2000
    mems = [make_semantic(f"big-{i}", big_fact) for i in range(20)]
    proc = make_procedural("proc-big", "B" * 1000)
    eps  = [make_episodic(f"ep-{i}", "C" * 500) for i in range(5)]

    result = QueryResult(semantic=mems, procedural=proc, episodic=eps)
    ctx = assemble_context(result)

    actual_tokens = _estimate_tokens(ctx.context_string)
    assert actual_tokens <= BUDGET_HARD_CEILING, (
        f"Context exceeded hard ceiling: {actual_tokens} > {BUDGET_HARD_CEILING}"
    )


def test_assembler_memory_ids_collected():
    """AssembledContext must return postgres_ids of all injected memories."""
    proc = make_procedural("proc-1", "ETL trigger")
    sem  = [make_semantic("sem-1", "Fact A"), make_semantic("sem-2", "Fact B")]
    eps  = [make_episodic("ep-1", "Task prompt")]

    result = QueryResult(semantic=sem, procedural=proc, episodic=eps)
    ctx = assemble_context(result)

    assert "proc-1" in ctx.memory_ids_used
    assert "sem-1"  in ctx.memory_ids_used
    assert "ep-1"   in ctx.memory_ids_used


def test_assembler_empty_result_returns_empty_string():
    """No memories found should return empty context string gracefully."""
    result = QueryResult(semantic=[], procedural=None, episodic=[])
    ctx = assemble_context(result)

    assert ctx.context_string == ""
    assert ctx.tokens_used == 0
    assert ctx.procedural_found is False
    assert ctx.semantic_count == 0
    assert ctx.episodic_count == 0