"""
scripts/seed_schema.py

Creates all three Weaviate collections + seeds Postgres task_types.
Must be run ONCE before any memory writes.
Safe to re-run — skips collections/rows that already exist.

Usage:
    python scripts/seed_schema.py

Validates against:
  - Weaviate HNSW params (spec Day 3 — validated constants)
  - Postgres task_types table (spec Day 2 — default task types)
  - Initial tenants: __global__ + agent_test
"""

import os
import sys

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    VectorDistances,
)
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEAVIATE_URL  = os.environ.get("WEAVIATE_URL",  "http://localhost:8080")
DATABASE_URL  = os.environ.get(
    "DATABASE_URL",
    "postgresql://memory:memory@localhost:5432/memory_engine"
)

# ---------------------------------------------------------------------------
# Weaviate collections
# Spec Day 3 validated HNSW params:
#   semantic:   ef=128, efC=256, maxConnections=64
#   procedural: ef=64,  efC=256, maxConnections=64
#   episodic:   ef=64,  efC=128, maxConnections=64
# ---------------------------------------------------------------------------
COLLECTIONS = {
    "SemanticMemory": {
        "ef":              128,
        "ef_construction": 256,
        "max_connections": 64,
        "properties": [
            Property(name="fact",             data_type=DataType.TEXT),
            Property(name="fact_type",        data_type=DataType.TEXT),
            Property(name="entities",         data_type=DataType.TEXT_ARRAY),
            Property(name="confidence",       data_type=DataType.NUMBER),
            Property(name="agent_id",         data_type=DataType.TEXT),
            Property(name="scope",            data_type=DataType.TEXT),
            Property(name="importance_score", data_type=DataType.NUMBER),
            Property(name="postgres_id",      data_type=DataType.UUID),
        ],
    },
    "ProceduralMemory": {
        "ef":              64,
        "ef_construction": 256,
        "max_connections": 64,
        "properties": [
            Property(name="trigger_condition", data_type=DataType.TEXT),
            Property(name="task_type",         data_type=DataType.TEXT),
            Property(name="confidence",        data_type=DataType.NUMBER),
            Property(name="agent_id",          data_type=DataType.TEXT),
            Property(name="importance_score",  data_type=DataType.NUMBER),
            Property(name="postgres_id",       data_type=DataType.UUID),
        ],
    },
    "EpisodicMemory": {
        "ef":              64,
        "ef_construction": 128,
        "max_connections": 64,
        "properties": [
            Property(name="task_prompt",      data_type=DataType.TEXT),
            Property(name="task_type",        data_type=DataType.TEXT),
            Property(name="outcome",          data_type=DataType.TEXT),
            Property(name="agent_id",         data_type=DataType.TEXT),
            Property(name="session_start",    data_type=DataType.DATE),
            Property(name="importance_score", data_type=DataType.NUMBER),
            Property(name="postgres_id",      data_type=DataType.UUID),
        ],
    },
}

# Initial tenants — created at schema time
# Add more via API as new agents register
INITIAL_TENANTS = [
    wvc.tenants.Tenant(name="__global__"),   # reserved — cross-agent facts
    wvc.tenants.Tenant(name="agent_test"),   # smoke-test tenant
]

# ---------------------------------------------------------------------------
# Default task_types (Postgres) — spec Day 2
# Add project-specific types at deploy time
# ---------------------------------------------------------------------------
DEFAULT_TASK_TYPES = [
    ("code_generation",    "Generate or modify source code"),
    ("code_review",        "Review and critique existing code"),
    ("debugging",          "Diagnose and fix software defects"),
    ("research",           "Gather and synthesize information"),
    ("data_analysis",      "Analyse datasets and produce insights"),
    ("writing",            "Produce or edit written content"),
    ("planning",           "Break down goals into actionable steps"),
    ("question_answering", "Answer factual or reasoning questions"),
    ("tool_use",           "Invoke external tools or APIs"),
    ("general",            "Catch-all for unclassified tasks"),
]


# ---------------------------------------------------------------------------
# Weaviate helpers
# ---------------------------------------------------------------------------
def create_collection(
    client: weaviate.WeaviateClient,
    name:   str,
    cfg:    dict,
) -> None:
    if client.collections.exists(name):
        print(f"  [SKIP] {name} already exists")
        # Still verify tenants exist (idempotent)
        _ensure_tenants(client, name)
        return

    client.collections.create(
        name=name,
        multi_tenancy_config=Configure.multi_tenancy(enabled=True),
        vectorizer_config=Configure.Vectorizer.none(),      # BYO embeddings
        vector_index_config=Configure.VectorIndex.hnsw(
            ef=cfg["ef"],
            ef_construction=cfg["ef_construction"],
            max_connections=cfg["max_connections"],
            distance_metric=VectorDistances.COSINE,
        ),
        properties=cfg["properties"],
    )
    print(f"  [OK]   {name} created")
    _ensure_tenants(client, name)


def _ensure_tenants(
    client: weaviate.WeaviateClient,
    name:   str,
) -> None:
    """Add initial tenants if not already present."""
    collection     = client.collections.get(name)
    existing       = {t.name for t in collection.tenants.get().values()}
    to_add         = [t for t in INITIAL_TENANTS if t.name not in existing]
    if to_add:
        collection.tenants.create(to_add)
        print(f"         Tenants added: {[t.name for t in to_add]}")
    else:
        print(f"         Tenants already present")


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------
def seed_task_types(engine) -> None:
    print("\nSeeding Postgres task_types ...")
    with engine.connect() as conn:
        inserted = 0
        skipped  = 0
        for name, description in DEFAULT_TASK_TYPES:
            result = conn.execute(text("""
                INSERT INTO task_types (id, name, description, created_at)
                VALUES (gen_random_uuid(), :name, :desc, NOW())
                ON CONFLICT (name) DO NOTHING
            """), {"name": name, "desc": description})
            if result.rowcount:
                inserted += 1
            else:
                skipped += 1
        conn.commit()
    print(f"  [OK]   inserted={inserted} skipped={skipped}")


def verify_postgres(engine) -> bool:
    """Verify all 7 required tables exist."""
    required = [
        "agents", "task_types", "sessions", "memory_entries",
        "memory_access_log", "memory_health_reports", "deletion_audit_log",
    ]
    print("\nVerifying Postgres tables ...")
    ok = True
    with engine.connect() as conn:
        for table in required:
            row = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": table}).scalar()
            status = "OK" if row else "MISSING"
            if not row:
                ok = False
            print(f"  [{status}] {table}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    errors = []

    # ---- Weaviate ----
    print(f"Connecting to Weaviate at {WEAVIATE_URL} ...")
    try:
        client = weaviate.connect_to_local(host="localhost", port=8080)
    except Exception as e:
        print(f"[ERROR] Cannot connect to Weaviate: {e}")
        print("        Is the container running? docker compose ps")
        sys.exit(1)

    try:
        print("Creating Weaviate collections ...\n")
        for name, cfg in COLLECTIONS.items():
            try:
                create_collection(client, name, cfg)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                errors.append(f"Weaviate/{name}: {e}")

        print("\nVerifying Weaviate collections ...")
        for name in COLLECTIONS:
            exists = client.collections.exists(name)
            status = "OK" if exists else "MISSING"
            if not exists:
                errors.append(f"Weaviate/{name} missing after create")
            print(f"  [{status}] {name}")

    finally:
        client.close()

    # ---- Postgres ----
    print(f"\nConnecting to Postgres ...")
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        pg_ok  = verify_postgres(engine)
        if not pg_ok:
            errors.append("Postgres: one or more tables missing — run ddl.sql first")
        else:
            seed_task_types(engine)
    except Exception as e:
        print(f"[ERROR] Postgres: {e}")
        errors.append(f"Postgres: {e}")

    # ---- Summary ----
    print("\n" + "="*50)
    if errors:
        print(f"[FAIL] Seed completed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[OK]  Seed complete — ready for first memory write.")


if __name__ == "__main__":
    main()