"""
Unit tests for the importance score formula.
"""

from core.lifecycle.scorer import (
    recency_score,
    access_score,
    outcome_score,
    compute_importance_score,
)
from datetime import datetime, timedelta


def test_recency_no_access_decays_fast():
    score = recency_score(days_since_confirmed=7, access_count=0)
    assert score < 0.01, f"Expected near-zero after 7 days with no access, got {score}"


def test_recency_high_access_decays_slow():
    score = recency_score(days_since_confirmed=7, access_count=20)
    assert score > 0.45, f"Expected >0.60 with 20 accesses at 7 days, got {score}"


def test_outcome_neutral_below_threshold():
    assert outcome_score(0, 0) == 0.5
    assert outcome_score(2, 2) == 0.5  # below 3 uses


def test_outcome_real_rate_above_threshold():
    assert outcome_score(8, 10) == 0.8


def test_importance_episodic_recency_dominant():
    # Episodic: recency weight 0.50 — old memory should score low
    now = datetime.utcnow()
    old_confirmed = now - timedelta(days=60)
    score = compute_importance_score(
        last_confirmed=old_confirmed,
        access_count=2,
        successful_uses=2,
        total_uses=2,
        explicit_signal=0.0,
        memory_type="episodic",
        base_importance=0.38,
    )
    assert score < 0.30, f"Old episodic should score low, got {score}"


def test_importance_procedural_outcome_dominant():
    # Procedural: outcome weight 0.40 — high success rate should score well
    now = datetime.utcnow()
    old_confirmed = now - timedelta(days=90)
    score = compute_importance_score(
        last_confirmed=old_confirmed,
        access_count=30,
        successful_uses=28,
        total_uses=30,
        explicit_signal=0.0,
        memory_type="procedural",
        base_importance=0.75,
    )
    assert score > 0.50, f"High-success procedural should score well even if old, got {score}"
