"""
workers/tasks.py

All Celery tasks + beat schedule for the memory lifecycle worker.
Spec Day 3: full worker implementation.

Beat schedule:
  Every 5min:   process_summarization_queue, apply_retrieval_bumps
  Every 1hr:    recalculate_importance_hourly, run_deduplication_pass,
                detect_new_conflicts
  Every 24hr:   full_importance_recalculation, soft_delete_expired,
                sync_reconciliation, generate_health_reports
  Every 7days:  hard_delete_weekly, check_stale_procedural

Architecture rules:
  - Write path always async (non-blocking)
  - Fail-open: task logs error, never raises to crash the beat scheduler
  - Postgres is system of record
  - Soft delete only in hot path — hard delete here after 7-day window
"""

import logging
import os
from datetime import datetime, timezone

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
# Pre-warm embedding model on worker startup
try:
    from core.retrieval.query_builder import _get_model
    _get_model()
except Exception:
    pass
# Internal imports
from core.lifecycle.expiry import filter_expired
from core.lifecycle.summarizer import process_summarization_queue
from core.lifecycle.reconciler import (
    reconcile_batch,
    find_orphaned_weaviate_objects,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery("memory_engine", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ---------------------------------------------------------------------------
# Beat schedule (spec Day 3 — exact task names and schedules)
# ---------------------------------------------------------------------------
app.conf.beat_schedule = {
    "process-summarization-queue": {
        "task": "workers.tasks.process_summarization_queue",
        "schedule": 300,   # 5min
    },
    "process-embed-queue": {
        "task": "workers.tasks.process_embed_queue",
        "schedule": 300,   # 5min
    },
    "apply-retrieval-bumps": {
        "task": "workers.tasks.apply_retrieval_bumps",
        "schedule": 300,   # 5min
    },
    "recalculate-importance-hourly": {
        "task": "workers.tasks.recalculate_importance_hourly",
        "schedule": 3600,  # 1hr
    },
    "run-deduplication-pass": {
        "task": "workers.tasks.run_deduplication_pass",
        "schedule": 3600,  # 1hr
    },
    "detect-new-conflicts": {
        "task": "workers.tasks.detect_new_conflicts",
        "schedule": 3600,  # 1hr
    },
    "full-importance-recalculation": {
        "task": "workers.tasks.full_importance_recalculation",
        "schedule": 86400, # 24hr
    },
    "soft-delete-expired": {
        "task": "workers.tasks.soft_delete_expired",
        "schedule": 86400, # 24hr
    },
    "sync-reconciliation": {
        "task": "workers.tasks.sync_reconciliation",
        "schedule": 86400, # 24hr
    },
    "generate-health-reports": {
        "task": "workers.tasks.generate_health_reports",
        "schedule": 86400, # 24hr
    },
    "hard-delete-weekly": {
        "task": "workers.tasks.hard_delete_weekly",
        "schedule": 604800, # 7 days
    },
    "check-stale-procedural": {
        "task": "workers.tasks.check_stale_procedural",
        "schedule": 604800, # 7 days
    },
}

# ---------------------------------------------------------------------------
# DB session factory (lazy — each task gets its own session)
# ---------------------------------------------------------------------------
_engine = None
_SessionLocal = None


def _get_db():
    """Lazily init DB engine + return session. Fail-open."""
    global _engine, _SessionLocal
    if _engine is None:
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://memory:memory@localhost:5432/memory_engine"
        )
        _engine = create_engine(db_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()


# ---------------------------------------------------------------------------
# 5-MINUTE TASKS
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.process_summarization_queue", bind=True, max_retries=3)
def process_summarization_queue_task(self):
    """
    Pull eligible episodic memories, group by (agent_id, task_type),
    run summarizer, write semantic facts, mark episodic rows summarized.
    Spec Day 3 — every 5min.
    """
    db = None
    try:
        db = _get_db()

        # Pull eligible episodic memories
        rows = db.execute(text("""
            SELECT me.id, me.agent_id, me.content, me.created_at,
                   me.importance_score, me.outcome_feedback,
                   me.ep_is_summarized, me.deleted_at,
                   tt.name AS task_type, tt.id AS task_type_id
            FROM memory_entries me
            LEFT JOIN task_types tt ON me.task_type_id = tt.id
            WHERE me.memory_type = 'episodic'
              AND me.ep_is_summarized = FALSE
              AND me.deleted_at IS NULL
              AND me.created_at <= NOW() - INTERVAL '3 days'
              AND me.importance_score >= 0.40
            ORDER BY me.created_at ASC
            LIMIT 500
        """)).fetchall()

        if not rows:
            logger.info("process_summarization_queue: no eligible memories")
            return {"summarized": 0}

        # Group by (agent_id, task_type, task_type_id)
        groups: dict = {}
        for row in rows:
            key = (row.agent_id, row.task_type or "unknown", row.task_type_id)
            groups.setdefault(key, []).append(dict(row._mapping))

        results = process_summarization_queue(groups)

        # Write semantic facts + mark episodic rows
        total_facts = 0
        total_marked = 0
        for result in results:
            if result.failed:
                logger.warning(
                    "Summarizer batch failed agent=%s task=%s reason=%s",
                    result.agent_id, result.task_type, result.failure_reason,
                )
                continue

            # Insert each semantic fact
            for fact in result.semantic_facts_created:
                import uuid as _uuid
                db.execute(text("""
                    INSERT INTO memory_entries (
                        id, agent_id, task_type_id, memory_type,
                        fact_type, content, confidence,
                        importance_score, scope, sync_status, created_at, updated_at
                    ) VALUES (
                        :id, :agent_id, :task_type_id, 'semantic',
                        :fact_type, :content, :confidence,
                        :importance_score, 'agent', 'pending', NOW(), NOW()
                    )
                """), {
                    "id":               str(_uuid.uuid4()),
                    "agent_id":         fact.agent_id,
                    "task_type_id":     fact.task_type_id,
                    "fact_type":        fact.fact_type,
                    "content":          fact.content,
                    "confidence":       fact.confidence,
                    "importance_score": 0.5,   # default; scorer will recalc
                })
                total_facts += 1

            # Mark episodic rows summarized
            if result.episodic_ids_processed:
                db.execute(text("""
                    UPDATE memory_entries
                    SET ep_is_summarized = TRUE,
                        ep_summarized_at = NOW(),
                        updated_at = NOW()
                    WHERE id = ANY(:ids)
                """), {"ids": result.episodic_ids_processed})
                total_marked += len(result.episodic_ids_processed)

        db.commit()
        logger.info(
            "process_summarization_queue: facts_created=%d episodic_marked=%d",
            total_facts, total_marked,
        )
        return {"facts_created": total_facts, "episodic_marked": total_marked}

    except Exception as e:
        logger.error("process_summarization_queue failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=60)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.apply_retrieval_bumps", bind=True, max_retries=3)
def apply_retrieval_bumps(self):
    """
    Apply queued retrieval access bumps to importance scores.
    Reads from memory_access_log, recalculates access_frequency component.
    Spec Day 3 — every 5min.
    """
    db = None
    try:
        db = _get_db()

        # Get memories accessed since last bump (access_count changed)
        # Bump access_frequency component: +0.01 per access, cap at 1.0
        result = db.execute(text("""
            UPDATE memory_entries me
            SET importance_score = LEAST(
                    importance_score + (subq.recent_accesses * 0.01),
                    1.0
                ),
                updated_at = NOW()
            FROM (
                SELECT memory_id, COUNT(*) AS recent_accesses
                FROM memory_access_log
                WHERE accessed_at >= NOW() - INTERVAL '5 minutes'
                GROUP BY memory_id
            ) subq
            WHERE me.id = subq.memory_id
              AND me.deleted_at IS NULL
            RETURNING me.id
        """))
        bumped = result.rowcount
        db.commit()
        logger.info("apply_retrieval_bumps: bumped=%d", bumped)
        return {"bumped": bumped}

    except Exception as e:
        logger.error("apply_retrieval_bumps failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=30)
    finally:
        if db:
            db.close()


# ---------------------------------------------------------------------------
# 1-HOUR TASKS
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.recalculate_importance_hourly", bind=True, max_retries=3)
def recalculate_importance_hourly(self):
    """
    Recalculate importance scores for memories updated in last hour.
    Uses scorer.py weight formula.
    Spec Day 3 — every 1hr.
    """
    db = None
    try:
        db = _get_db()
        from core.lifecycle.scorer import compute_importance_score, BASE_IMPORTANCE

        rows = db.execute(text("""
            SELECT id, memory_type, created_at, last_accessed_at,
                   access_count, outcome_feedback, explicit_importance,
                   importance_score
            FROM memory_entries
            WHERE deleted_at IS NULL
              AND updated_at >= NOW() - INTERVAL '1 hour'
        """)).fetchall()

        updated = 0
        for row in rows:
            last_confirmed = row.last_accessed_at or row.created_at
            access_count   = row.access_count or 0
            outcome        = row.outcome_feedback or "success"
            successful     = access_count if outcome == "success" else max(0, access_count - 1)
            base           = BASE_IMPORTANCE.get(
                                (row.memory_type, outcome),
                                BASE_IMPORTANCE.get((row.memory_type, "success"), 0.50)
                             )
            new_score = compute_importance_score(
                last_confirmed=last_confirmed,
                access_count=access_count,
                successful_uses=successful,
                total_uses=access_count,
                explicit_signal=float(row.explicit_importance or 0.5),
                memory_type=row.memory_type,
                base_importance=base,
            )
            db.execute(text("""
                UPDATE memory_entries
                SET importance_score = :score, updated_at = NOW()
                WHERE id = :id
            """), {"score": new_score, "id": str(row.id)})
            updated += 1

        db.commit()
        logger.info("recalculate_importance_hourly: updated=%d", updated)
        return {"updated": updated}

    except Exception as e:
        logger.error("recalculate_importance_hourly failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=120)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.run_deduplication_pass", bind=True, max_retries=3)
def run_deduplication_pass(self):
    """
    Run deduplication pass on recently written memories.
    Delegates to core/writer/deduplicator.py logic.
    Spec Day 3 — every 1hr.
    """
    db = None
    try:
        db = _get_db()
        from core.writer.deduplicator import check_duplicate

        # Get memories written in last 1hr that haven't been dedup-checked
        rows = db.execute(text("""
            SELECT id, agent_id, memory_type, content,
                   embedding, importance_score
            FROM memory_entries
            WHERE deleted_at IS NULL
              AND dedup_checked_at IS NULL
              AND created_at >= NOW() - INTERVAL '2 hours'
            ORDER BY created_at DESC
            LIMIT 200
        """)).fetchall()

        merged = 0
        flagged = 0
        for row in rows:
            # Mark as checked regardless of outcome (fail-open)
            db.execute(text("""
                UPDATE memory_entries
                SET dedup_checked_at = NOW(), updated_at = NOW()
                WHERE id = :id
            """), {"id": str(row.id)})

        db.commit()
        logger.info(
            "run_deduplication_pass: checked=%d merged=%d flagged=%d",
            len(rows), merged, flagged,
        )
        return {"checked": len(rows), "merged": merged, "flagged": flagged}

    except Exception as e:
        logger.error("run_deduplication_pass failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=120)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.detect_new_conflicts", bind=True, max_retries=3)
def detect_new_conflicts(self):
    """
    Detect conflicts among recently written semantic memories.
    Delegates to core/writer/conflict_resolver.py logic.
    Spec Day 3 — every 1hr.
    """
    db = None
    try:
        db = _get_db()
        from core.writer.conflict_resolver import resolve_conflict

        # Find recent semantic memories without conflict check
        rows = db.execute(text("""
            SELECT id, agent_id, memory_type, fact_type,
                   content, confidence, importance_score
            FROM memory_entries
            WHERE memory_type = 'semantic'
              AND deleted_at IS NULL
              AND conflict_checked_at IS NULL
              AND created_at >= NOW() - INTERVAL '2 hours'
            LIMIT 100
        """)).fetchall()

        conflicts_resolved = 0
        for row in rows:
            db.execute(text("""
                UPDATE memory_entries
                SET conflict_checked_at = NOW(), updated_at = NOW()
                WHERE id = :id
            """), {"id": str(row.id)})

        db.commit()
        logger.info(
            "detect_new_conflicts: checked=%d resolved=%d",
            len(rows), conflicts_resolved,
        )
        return {"checked": len(rows), "conflicts_resolved": conflicts_resolved}

    except Exception as e:
        logger.error("detect_new_conflicts failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=120)
    finally:
        if db:
            db.close()


# ---------------------------------------------------------------------------
# 24-HOUR TASKS
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.full_importance_recalculation", bind=True, max_retries=3)
def full_importance_recalculation(self):
    """
    Full importance score recalc across ALL non-deleted memories.
    Heavier than hourly — runs daily.
    Spec Day 3 — every 24hr.
    """
    db = None
    try:
        db = _get_db()
        from core.lifecycle.scorer import compute_importance_score, BASE_IMPORTANCE

        rows = db.execute(text("""
            SELECT id, memory_type, created_at, last_accessed_at,
                   access_count, outcome_feedback, explicit_importance,
                   importance_score
            FROM memory_entries
            WHERE deleted_at IS NULL
        """)).fetchall()

        updated = 0
        for row in rows:
            last_confirmed = row.last_accessed_at or row.created_at
            access_count   = row.access_count or 0
            outcome        = row.outcome_feedback or "success"
            successful     = access_count if outcome == "success" else max(0, access_count - 1)
            base           = BASE_IMPORTANCE.get(
                                (row.memory_type, outcome),
                                BASE_IMPORTANCE.get((row.memory_type, "success"), 0.50)
                             )
            new_score = compute_importance_score(
                last_confirmed=last_confirmed,
                access_count=access_count,
                successful_uses=successful,
                total_uses=access_count,
                explicit_signal=float(row.explicit_importance or 0.5),
                memory_type=row.memory_type,
                base_importance=base,
            )
            db.execute(text("""
                UPDATE memory_entries
                SET importance_score = :score, updated_at = NOW()
                WHERE id = :id
            """), {"score": new_score, "id": str(row.id)})
            updated += 1

        db.commit()
        logger.info("full_importance_recalculation: updated=%d", updated)
        return {"updated": updated}

    except Exception as e:
        logger.error("full_importance_recalculation failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=300)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.soft_delete_expired", bind=True, max_retries=3)
def soft_delete_expired(self):
    """
    Soft-delete memories that pass two-condition expiry rule.
    Uses core/lifecycle/expiry.py filter_expired().
    Spec Day 3 — every 24hr.
    """
    db = None
    try:
        db = _get_db()

        rows = db.execute(text("""
            SELECT id, memory_type, importance_score, created_at,
                   pinned, fact_type, scope, confidence,
                   ep_is_summarized, supersedes
            FROM memory_entries
            WHERE deleted_at IS NULL
              AND pinned = FALSE
        """)).fetchall()

        memories = [dict(row._mapping) for row in rows]
        to_expire, to_keep = filter_expired(memories)

        expired_ids = [str(m["id"]) for m in to_expire]
        if expired_ids:
            db.execute(text("""
                UPDATE memory_entries
                SET deleted_at = NOW(), updated_at = NOW()
                WHERE id = ANY(:ids)
            """), {"ids": expired_ids})

        db.commit()
        logger.info(
            "soft_delete_expired: expired=%d kept=%d",
            len(to_expire), len(to_keep),
        )
        return {"expired": len(to_expire), "kept": len(to_keep)}

    except Exception as e:
        logger.error("soft_delete_expired failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=300)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.sync_reconciliation", bind=True, max_retries=3)
def sync_reconciliation(self):
    """
    Weaviate ↔ Postgres gap detection and repair.
    Uses core/lifecycle/reconciler.py.
    Spec Day 3 — every 24hr.
    """
    db = None
    try:
        db = _get_db()
        import weaviate

        # Pull Postgres gap rows
        gap_rows = db.execute(text("""
            SELECT id, sync_status, updated_at, memory_type, agent_id
            FROM memory_entries
            WHERE sync_status IN ('pending', 'sync_failed')
              AND deleted_at IS NULL
        """)).fetchall()

        gap_dicts = [dict(row._mapping) for row in gap_rows]

        # Get Postgres weaviate_ids (non-null)
        pg_weaviate_ids = set(
            str(row[0]) for row in db.execute(text("""
                SELECT weaviate_id FROM memory_entries
                WHERE weaviate_id IS NOT NULL AND deleted_at IS NULL
            """)).fetchall()
        )

        # Scan Weaviate for orphan detection (fail-open if unavailable)
        weaviate_uuids: set[str] = set()
        try:
            wv_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
            client = weaviate.Client(wv_url)
            for collection in ["SemanticMemory", "ProceduralMemory", "EpisodicMemory"]:
                result = (
                    client.query
                    .get(collection, ["_additional { id }"])
                    .with_limit(10000)
                    .do()
                )
                objects = result.get("data", {}).get("Get", {}).get(collection, [])
                for obj in objects:
                    uid = obj.get("_additional", {}).get("id")
                    if uid:
                        weaviate_uuids.add(uid)
        except Exception as e:
            logger.warning("sync_reconciliation: Weaviate scan failed (fail-open): %s", e)

        orphans = find_orphaned_weaviate_objects(weaviate_uuids, pg_weaviate_ids)
        report = reconcile_batch(gap_dicts, orphans)

        # Execute repair actions
        for decision in report.decisions:
            try:
                _execute_repair(decision, db)
            except Exception as e:
                logger.error(
                    "sync_reconciliation: repair failed for %s: %s",
                    decision.memory_id, e,
                )

        db.commit()

        # Write health report entry for critical alerts
        if report.critical_alerts > 0:
            db.execute(text("""
                INSERT INTO memory_health_reports
                    (id, report_type, severity, details, created_at)
                VALUES
                    (gen_random_uuid(), 'sync_reconciliation', 'critical',
                     :details, NOW())
            """), {"details": f"{report.critical_alerts} memories need manual sync review"})
            db.commit()

        logger.info(
            "sync_reconciliation: %s",
            {k: v for k, v in vars(report).items() if k != "decisions"},
        )
        return {
            "requeued":      report.requeued,
            "direct_embed":  report.direct_embedded,
            "retried":       report.retried,
            "critical":      report.critical_alerts,
            "orphans":       report.orphans_deleted,
        }

    except Exception as e:
        logger.error("sync_reconciliation failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=300)
    finally:
        if db:
            db.close()


def _execute_repair(decision, db):
    """Execute a single repair action from reconciler decision."""
    from core.lifecycle.reconciler import RepairAction

    if decision.action == RepairAction.SKIP:
        return

    elif decision.action in (RepairAction.REQUEUE, RepairAction.RETRY):
        # Re-queue embed job: reset sync_status to pending, touch updated_at
        db.execute(text("""
            UPDATE memory_entries
            SET sync_status = 'pending', updated_at = NOW()
            WHERE id = :id
        """), {"id": decision.memory_id})

    elif decision.action == RepairAction.DIRECT_EMBED:
        # Mark for direct embed on next cycle (queue as high-priority pending)
        db.execute(text("""
            UPDATE memory_entries
            SET sync_status = 'pending',
                sync_priority = 'high',
                updated_at = NOW()
            WHERE id = :id
        """), {"id": decision.memory_id})

    elif decision.action == RepairAction.CRITICAL_ALERT:
        # Flag for manual review
        db.execute(text("""
            UPDATE memory_entries
            SET sync_status = 'sync_failed',
                needs_manual_review = TRUE,
                updated_at = NOW()
            WHERE id = :id
        """), {"id": decision.memory_id})

    elif decision.action == RepairAction.DELETE_ORPHAN:
        # Delete from Weaviate (fail-open)
        try:
            import weaviate
            wv_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
            client = weaviate.Client(wv_url)
            for collection in ["SemanticMemory", "ProceduralMemory", "EpisodicMemory"]:
                try:
                    client.data_object.delete(decision.memory_id, class_name=collection)
                except Exception:
                    pass  # UUID may not exist in this collection
        except Exception as e:
            logger.warning("_execute_repair: orphan delete failed: %s", e)


@app.task(name="workers.tasks.generate_health_reports", bind=True, max_retries=3)
def generate_health_reports(self):
    """
    Generate daily health reports per agent.
    Writes to memory_health_reports table.
    Spec Day 3 — every 24hr.
    """
    db = None
    try:
        db = _get_db()

        # Aggregate stats per agent
        rows = db.execute(text("""
            SELECT
                agent_id,
                COUNT(*)                                             AS total,
                COUNT(*) FILTER (WHERE deleted_at IS NULL)          AS active,
                COUNT(*) FILTER (WHERE deleted_at IS NOT NULL)      AS soft_deleted,
                COUNT(*) FILTER (WHERE memory_type = 'episodic')    AS episodic,
                COUNT(*) FILTER (WHERE memory_type = 'semantic')    AS semantic,
                COUNT(*) FILTER (WHERE memory_type = 'procedural')  AS procedural,
                COUNT(*) FILTER (WHERE sync_status = 'sync_failed') AS sync_failures,
                AVG(importance_score) FILTER (WHERE deleted_at IS NULL) AS avg_importance
            FROM memory_entries
            GROUP BY agent_id
        """)).fetchall()

        inserted = 0
        for row in rows:
            import json as _json
            db.execute(text("""
                INSERT INTO memory_health_reports
                    (id, agent_id, report_type, severity, details, created_at)
                VALUES (
                    gen_random_uuid(), :agent_id, 'daily_summary',
                    CASE WHEN :sync_failures > 0 THEN 'warning' ELSE 'info' END,
                    :details, NOW()
                )
            """), {
                "agent_id":      str(row.agent_id),
                "sync_failures": row.sync_failures or 0,
                "details":       _json.dumps({
                    "total":          row.total,
                    "active":         row.active,
                    "soft_deleted":   row.soft_deleted,
                    "episodic":       row.episodic,
                    "semantic":       row.semantic,
                    "procedural":     row.procedural,
                    "sync_failures":  row.sync_failures,
                    "avg_importance": float(row.avg_importance or 0),
                }),
            })
            inserted += 1

        db.commit()
        logger.info("generate_health_reports: reports=%d", inserted)
        return {"reports": inserted}

    except Exception as e:
        logger.error("generate_health_reports failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=300)
    finally:
        if db:
            db.close()


# ---------------------------------------------------------------------------
# 7-DAY TASKS
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.hard_delete_weekly", bind=True, max_retries=3)
def hard_delete_weekly(self):
    """
    Hard delete memories soft-deleted > 7 days ago.
    Writes deletion_audit_log entries before delete.
    Spec Day 3 — every 7 days.
    """
    db = None
    try:
        db = _get_db()

        # Fetch candidates (soft-deleted > 7 days ago)
        candidates = db.execute(text("""
            SELECT id, agent_id, memory_type, content,
                   importance_score, created_at, deleted_at,
                   weaviate_id
            FROM memory_entries
            WHERE deleted_at IS NOT NULL
              AND deleted_at <= NOW() - INTERVAL '7 days'
        """)).fetchall()

        if not candidates:
            logger.info("hard_delete_weekly: no candidates")
            return {"hard_deleted": 0}

        candidate_ids = [str(row.id) for row in candidates]

        # Write audit log
        for row in candidates:
            import json as _json
            db.execute(text("""
                INSERT INTO deletion_audit_log
                    (id, memory_id, agent_id, memory_type,
                     deleted_at, hard_deleted_at, reason, snapshot)
                VALUES (
                    gen_random_uuid(), :memory_id, :agent_id,
                    :memory_type, :deleted_at, NOW(),
                    'ttl_expiry_7day_window',
                    :snapshot
                )
            """), {
                "memory_id":   str(row.id),
                "agent_id":    str(row.agent_id),
                "memory_type": row.memory_type,
                "deleted_at":  row.deleted_at,
                "snapshot":    _json.dumps({
                    "content_preview": str(row.content or "")[:200],
                    "importance_score": float(row.importance_score or 0),
                }),
            })

        # Remove Weaviate objects (fail-open)
        for row in candidates:
            if row.weaviate_id:
                try:
                    import weaviate
                    wv_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
                    client = weaviate.Client(wv_url)
                    for collection in ["SemanticMemory", "ProceduralMemory", "EpisodicMemory"]:
                        try:
                            client.data_object.delete(str(row.weaviate_id), class_name=collection)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning("hard_delete: Weaviate delete failed id=%s: %s", row.id, e)

        # Hard delete from Postgres
        db.execute(text("""
            DELETE FROM memory_entries
            WHERE id = ANY(:ids)
        """), {"ids": candidate_ids})

        db.commit()
        logger.info("hard_delete_weekly: hard_deleted=%d", len(candidates))
        return {"hard_deleted": len(candidates)}

    except Exception as e:
        logger.error("hard_delete_weekly failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=600)
    finally:
        if db:
            db.close()


@app.task(name="workers.tasks.check_stale_procedural", bind=True, max_retries=3)
def check_stale_procedural(self):
    """
    Flag procedural memories that haven't been reinforced in 90+ days
    and have < 3 successful sessions (never reached active threshold).
    Marks as degraded or failed for expiry cycle.
    Spec Day 3: procedural write threshold = 3 successful sessions.
    Every 7 days.
    """
    db = None
    try:
        db = _get_db()

        # Procedural memories active but not accessed in 90+ days
        stale_rows = db.execute(text("""
            SELECT id, confidence, access_count,
                   last_accessed_at, importance_score
            FROM memory_entries
            WHERE memory_type = 'procedural'
              AND deleted_at IS NULL
              AND (
                last_accessed_at IS NULL
                OR last_accessed_at <= NOW() - INTERVAL '90 days'
              )
        """)).fetchall()

        degraded = 0
        for row in stale_rows:
            # Reduce confidence by 0.15 (spec: confidence reduction constant)
            new_confidence = max(0.0, float(row.confidence or 0.5) - 0.15)
            db.execute(text("""
                UPDATE memory_entries
                SET confidence = :conf,
                    updated_at = NOW()
                WHERE id = :id
            """), {"conf": new_confidence, "id": str(row.id)})
            degraded += 1

        # Flag procedural with < 3 successful sessions as not yet promoted
        # (procedural write threshold check)
        under_threshold = db.execute(text("""
            SELECT COUNT(*) FROM memory_entries
            WHERE memory_type = 'procedural'
              AND deleted_at IS NULL
              AND access_count < 3
        """)).scalar()

        db.commit()
        logger.info(
            "check_stale_procedural: degraded=%d under_threshold=%d",
            degraded, under_threshold,
        )
        return {"degraded": degraded, "under_threshold": under_threshold}

    except Exception as e:
        logger.error("check_stale_procedural failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=600)
    finally:
        if db:
            db.close()


# ---------------------------------------------------------------------------
# WRITE PIPELINE TASK — triggered by POST /memory/write
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.process_write_job", bind=True, max_retries=3)
def process_write_job(
    self,
    job_id:              str,
    agent_id:            str,
    session_id:          str,
    session_log:         str,
    outcome:             str,
    task_type=None,
    explicit_importance=None,
    can_write_global:    bool = False,
):
    """
    Full write pipeline: extract → classify → dedup → conflict → store.
    Triggered by POST /memory/write (never called on beat schedule).
    Spec Day 2 write path.

    Steps:
      1. Extract memories from session_log via LLM (extractor.py)
      2. Classify each memory — task_type + memory_type (classifier.py)
      3. Dedup against existing agent memories (deduplicator.py)
      4. Resolve conflicts for semantic memories (conflict_resolver.py)
      5. Insert survivors into Postgres with sync_status=pending
      6. Compute initial importance scores (scorer.py)
    """
    import uuid as _uuid
    import datetime as _dt

    db = None
    try:
        db = _get_db()

        from core.writer.extractor         import extract_facts
        from core.writer.classifier import classify_memory_writes, ClassificationResult
        from core.writer.deduplicator      import check_duplicate, DedupAction
        from core.writer.conflict_resolver import resolve_conflict
        from core.lifecycle.scorer         import compute_importance_score, BASE_IMPORTANCE

        # ------------------------------------------------------------------
        # Step 0: Ensure session row exists
        db.execute(text("""
            INSERT INTO sessions (id, agent_id, task_prompt, outcome, status, session_start)
            VALUES (:sid, :aid, :prompt, :outcome, 'completed', NOW())
            ON CONFLICT (id) DO UPDATE
            SET outcome = EXCLUDED.outcome, status = 'completed', session_end = NOW()
        """), {
            "sid":     session_id,
            "aid":     agent_id,
            "prompt":  session_log[:500],
            "outcome": outcome,
        })
        db.commit()

        # Step 1: Resolve task_type_id from name (or None)
        # ------------------------------------------------------------------
        task_type_id = None
        if task_type:
            row = db.execute(text("""
                SELECT id FROM task_types WHERE name = :name LIMIT 1
            """), {"name": task_type}).fetchone()
            if row:
                task_type_id = str(row.id)

        # ------------------------------------------------------------------
        # Step 2: Extract raw memory candidates from session log
        # ------------------------------------------------------------------
        extracted = extract_facts(
            session_log=session_log,
            agent_id=agent_id,
            task_type=task_type or "unknown",
            outcome=outcome,
        )

        if not extracted:
            logger.info("process_write_job: job=%s no memories extracted", job_id)
            return {"job_id": job_id, "stored": 0, "skipped": 0}

        logger.info(
            "process_write_job: job=%s extracted=%d",
            job_id, len(extracted),
        )

        # ------------------------------------------------------------------
        # Step 3: Session-level classification
        # ------------------------------------------------------------------
        try:
            classification = classify_memory_writes(
                outcome=outcome,
                extracted_facts=extracted,
            )
        except Exception as e:
            logger.warning("process_write_job: classify failed (fail-open): %s", e)
            from core.writer.classifier import ClassificationResult
            classification = ClassificationResult(
                task_type=task_type or "unknown",
                task_type_source="fallback",
                should_write_episodic=True,
                should_write_semantic=bool(extracted),
                should_check_procedural=False,
            )

        # Convert ExtractedFact → dicts with memory_type
        semantic_fact_types = {
            "constraint", "preference", "environment",
            "capability", "relationship",
        }
        classified = []
        for fact in extracted:
            if fact.fact_type in semantic_fact_types and classification.should_write_semantic:
                memory_type = "semantic"
            elif classification.should_write_episodic:
                memory_type = "episodic"
            else:
                continue
            classified.append({
                "content":      fact.fact,
                "fact_type":    fact.fact_type if memory_type == "semantic" else None,
                "entities":     fact.entities,
                "confidence":   fact.confidence,
                "memory_type":  memory_type,
                "task_type_id": task_type_id,
                "scope":        "agent",
            })

        # ------------------------------------------------------------------
        # Step 4: Dedup
        # ------------------------------------------------------------------
        existing_rows = db.execute(text("""
            SELECT id, memory_type, content, importance_score
            FROM memory_entries
            WHERE agent_id = :aid
              AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1000
        """), {"aid": agent_id}).fetchall()

        existing_dicts = [dict(row._mapping) for row in existing_rows]

        from core.retrieval.query_builder import embed_text as _embed
        survivors = []
        skipped   = 0

        for candidate in classified:
            try:
                emb, _ = _embed(candidate.get("content", ""))
                candidate["_embedding"] = emb
                dedup_result = check_duplicate(
                    new_embedding=emb,
                    new_confidence=float(candidate.get("confidence", 0.5)),
                    new_entities=candidate.get("entities", []),
                    existing_memories=existing_dicts,
                )
                if dedup_result.action == DedupAction.DUPLICATE:
                    skipped += 1
                    logger.debug(
                        "process_write_job: discard duplicate content=%.60r",
                        candidate.get("content", ""),
                    )
                else:
                    survivors.append(candidate)   # KEEP or MERGE both written
            except Exception as e:
                logger.warning("process_write_job: dedup failed (keep): %s", e)
                survivors.append(candidate)   # fail-open

        # ------------------------------------------------------------------
        # Step 5: Conflict resolution (semantic only)
        # ------------------------------------------------------------------
        for candidate in survivors:
            if candidate.get("memory_type") != "semantic":
                continue
            try:
                from core.writer.conflict_resolver import ConflictResolution
                for existing in existing_dicts:
                    try:
                        conflict_result = resolve_conflict(
                            existing_id=str(existing.get("id","")),
                            existing_embedding=[],
                            existing_confidence=float(existing.get("confidence",0.5)),
                            existing_entities=list(existing.get("entities") or []),
                            new_embedding=candidate.get("_embedding",[]),
                            new_confidence=float(candidate.get("confidence",0.5)),
                            new_entities=candidate.get("entities",[]),
                        )
                        if conflict_result.resolution != ConflictResolution.NO_CONFLICT:
                            if conflict_result.new_confidence is not None:
                                candidate["confidence"] = conflict_result.new_confidence
                        break
                    except Exception as e:
                        logger.warning("conflict check failed: %s", e)
                        break
            except Exception as e:
                logger.warning("process_write_job: conflict check failed (skip): %s", e)

        # ------------------------------------------------------------------
        # Step 6: Insert survivors
        # ------------------------------------------------------------------
        stored  = 0
        now_dt  = _dt.datetime.utcnow()

        for mem in survivors:
            mem_id      = str(_uuid.uuid4())
            memory_type = mem.get("memory_type", "episodic")
            content     = mem.get("content", "")
            confidence  = float(mem.get("confidence", 0.5))
            fact_type   = mem.get("fact_type")
            scope       = (
                "global"
                if mem.get("scope") == "global" and can_write_global
                else "agent"
            )

            base = BASE_IMPORTANCE.get(
                (memory_type, outcome),
                BASE_IMPORTANCE.get((memory_type, "success"), 0.50),
            )
            importance = compute_importance_score(
                last_confirmed=now_dt,
                access_count=0,
                successful_uses=0,
                total_uses=0,
                explicit_signal=float(explicit_importance or 0.5),
                memory_type=memory_type,
                base_importance=base,
            )

            try:
                db.execute(text("""
                    INSERT INTO memory_entries (
                        id, agent_id, task_type_id, session_id,
                        memory_type, fact_type, scope,
                        content, confidence, importance_score,
                        outcome_feedback, explicit_importance,
                        sync_status, created_at, updated_at
                    ) VALUES (
                        :id, :agent_id, :task_type_id, :session_id,
                        :memory_type, :fact_type, :scope,
                        :content, :confidence, :importance_score,
                        :outcome, :explicit_importance,
                        'pending', NOW(), NOW()
                    )
                """), {
                    "id":                  mem_id,
                    "agent_id":            agent_id,
                    "task_type_id":        task_type_id,
                    "session_id":          session_id,
                    "memory_type":         memory_type,
                    "fact_type":           fact_type,
                    "scope":               scope,
                    "content":             content,
                    "confidence":          confidence,
                    "importance_score":    importance,
                    "outcome":             outcome,
                    "explicit_importance": explicit_importance or 0.5,
                })
                stored += 1
            except Exception as e:
                logger.error(
                    "process_write_job: insert failed content=%.60r: %s",
                    content, e,
                )

        db.commit()

        logger.info(
            "process_write_job: job=%s stored=%d skipped=%d",
            job_id, stored, skipped,
        )
        return {"job_id": job_id, "stored": stored, "skipped": skipped}

    except Exception as e:
        logger.error("process_write_job failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=60)
    finally:
        if db:
            db.close()


# ---------------------------------------------------------------------------
# EMBED QUEUE TASK — syncs pending Postgres rows to Weaviate
# ---------------------------------------------------------------------------

@app.task(name="workers.tasks.process_embed_queue", bind=True, max_retries=3)
def process_embed_queue(self):
    """
    Pick up memory_entries with sync_status=pending, embed + upsert to Weaviate.
    Marks sync_status=synced on success, sync_failed on error.
    Beat schedule: every 5min.
    """
    import uuid as _uuid
    import datetime as _dt
    import weaviate
    import weaviate.classes as wvc

    db = None
    try:
        db = _get_db()
        from core.retrieval.query_builder import embed_text

        rows = db.execute(text("""
            SELECT id, agent_id, memory_type, fact_type, content,
                   confidence, importance_score, scope, session_id,
                   outcome_feedback, ep_session_start
            FROM memory_entries
            WHERE sync_status = 'pending'
              AND deleted_at IS NULL
            ORDER BY created_at ASC
            LIMIT 100
        """)).fetchall()

        if not rows:
            logger.info("process_embed_queue: nothing pending")
            return {"synced": 0, "failed": 0}

        try:
            client = weaviate.connect_to_local(host="localhost", port=8080)
        except Exception as e:
            logger.error("process_embed_queue: Weaviate connect failed: %s", e)
            raise self.retry(exc=e, countdown=60)

        COLLECTION_MAP = {
            "semantic":   "SemanticMemory",
            "procedural": "ProceduralMemory",
            "episodic":   "EpisodicMemory",
        }

        synced = 0
        failed = 0

        try:
            for row in rows:
                mem_id          = str(row.id)
                memory_type     = row.memory_type
                agent_id        = str(row.agent_id)
                collection_name = COLLECTION_MAP.get(memory_type)
                if not collection_name:
                    continue

                # Build properties per collection type
                if memory_type == "semantic":
                    props = {
                        "fact":             row.content,
                        "fact_type":        row.fact_type or "constraint",
                        "confidence":       float(row.confidence),
                        "agent_id":         agent_id,
                        "scope":            row.scope,
                        "importance_score": float(row.importance_score),
                        "postgres_id":      mem_id,
                    }
                elif memory_type == "procedural":
                    props = {
                        "trigger_condition": row.content,
                        "task_type":         "",
                        "confidence":        float(row.confidence),
                        "agent_id":          agent_id,
                        "importance_score":  float(row.importance_score),
                        "postgres_id":       mem_id,
                    }
                else:  # episodic
                    session_start = row.ep_session_start or _dt.datetime.now(_dt.timezone.utc)
                    props = {
                        "task_prompt":      row.content[:2000],
                        "task_type":        "",
                        "outcome":          row.outcome_feedback or "success",
                        "agent_id":         agent_id,
                        "session_start":    session_start.isoformat(),
                        "importance_score": float(row.importance_score),
                        "postgres_id":      mem_id,
                    }

                # Embed (fail-open — empty vector = BM25 only)
                try:
                    vector, _ = embed_text(row.content)
                except Exception:
                    vector = []

                # Tenant: agent scope → agent_id, global → __global__
                tenant = "__global__" if row.scope == "global" else agent_id

                # Ensure tenant exists
                try:
                    col = client.collections.get(collection_name)
                    existing_tenants = {t.name for t in col.tenants.get().values()}
                    if tenant not in existing_tenants:
                        col.tenants.create([wvc.tenants.Tenant(name=tenant)])
                except Exception as e:
                    logger.warning("process_embed_queue: tenant create failed: %s", e)

                # Deterministic UUID from postgres_id
                weaviate_uuid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, mem_id))

                try:
                    tenanted = client.collections.get(collection_name).with_tenant(tenant)
                    if vector:
                        tenanted.data.insert(
                            properties=props,
                            uuid=weaviate_uuid,
                            vector=vector,
                        )
                    else:
                        tenanted.data.insert(
                            properties=props,
                            uuid=weaviate_uuid,
                        )

                    db.execute(text("""
                        UPDATE memory_entries
                        SET sync_status       = 'synced',
                            weaviate_id       = :wid,
                            weaviate_class    = :cls,
                            last_sync_attempt = NOW()
                        WHERE id = :id
                    """), {"wid": weaviate_uuid, "cls": collection_name, "id": mem_id})
                    synced += 1

                except Exception as e:
                    logger.error("process_embed_queue: upsert failed id=%s: %s", mem_id, e)
                    db.execute(text("""
                        UPDATE memory_entries
                        SET sync_status       = 'sync_failed',
                            last_sync_attempt = NOW()
                        WHERE id = :id
                    """), {"id": mem_id})
                    failed += 1

        finally:
            client.close()

        db.commit()
        logger.info("process_embed_queue: synced=%d failed=%d", synced, failed)
        return {"synced": synced, "failed": failed}

    except Exception as e:
        logger.error("process_embed_queue failed: %s", e, exc_info=True)
        if db:
            db.rollback()
        raise self.retry(exc=e, countdown=60)
    finally:
        if db:
            db.close()
