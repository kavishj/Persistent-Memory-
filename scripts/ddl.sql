-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================
-- ENUM TYPES (skip if already created)
-- ============================================================
DO $$ BEGIN
    CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE fact_type AS ENUM ('constraint', 'preference', 'environment', 'capability', 'relationship');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE memory_outcome AS ENUM ('success', 'failure', 'partial');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE memory_scope AS ENUM ('agent', 'global');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE sync_status AS ENUM ('pending', 'synced', 'sync_failed', 'archived');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE conflict_tag AS ENUM ('none', 'unresolved_conflict', 'possible_conflict', 'coarser_version', 'review_pending');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- TABLE: agents
-- ============================================================
CREATE TABLE IF NOT EXISTS agents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_key           TEXT NOT NULL UNIQUE,
    display_name        TEXT NOT NULL,
    scope_tier          TEXT NOT NULL DEFAULT 'standard'
                        CHECK (scope_tier IN ('standard', 'elevated', 'operator')),
    is_active           BOOLEAN NOT NULL DEFAULT true,
    api_key_hash        TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agents_agent_key ON agents(agent_key);
CREATE INDEX IF NOT EXISTS idx_agents_scope_tier ON agents(scope_tier);

-- ============================================================
-- TABLE: task_types
-- ============================================================
CREATE TABLE IF NOT EXISTS task_types (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                TEXT NOT NULL UNIQUE,
    description         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- TABLE: sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id            UUID NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
    task_type_id        UUID REFERENCES task_types(id),
    task_prompt         TEXT NOT NULL DEFAULT '',
    task_prompt_tokens  INTEGER,
    final_output        TEXT,
    outcome             memory_outcome,
    error_message       TEXT,
    duration_ms         INTEGER,
    token_cost          INTEGER,
    memories_injected   UUID[],
    session_start       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_end         TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'active',
    metadata            JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_task_type_id ON sessions(task_type_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_start ON sessions(session_start DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_outcome ON sessions(outcome);

-- ============================================================
-- TABLE: memory_entries
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_entries (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id            UUID REFERENCES agents(id) ON DELETE RESTRICT,
    scope               memory_scope NOT NULL DEFAULT 'agent',
    memory_type         memory_type NOT NULL,
    fact_type           fact_type,
    content             TEXT NOT NULL,
    content_tokens      INTEGER,
    entities            TEXT[],
    weaviate_class      TEXT,
    weaviate_id         UUID,
    sync_status         sync_status NOT NULL DEFAULT 'pending',
    sync_priority       TEXT NOT NULL DEFAULT 'normal',
    last_sync_attempt   TIMESTAMPTZ,
    importance_score    FLOAT NOT NULL DEFAULT 0.5
                        CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    base_importance     FLOAT NOT NULL DEFAULT 0.5
                        CHECK (base_importance >= 0.0 AND base_importance <= 1.0),
    confidence          FLOAT NOT NULL DEFAULT 0.5
                        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    access_count        INTEGER NOT NULL DEFAULT 0,
    successful_uses     INTEGER NOT NULL DEFAULT 0,
    total_uses          INTEGER NOT NULL DEFAULT 0,
    explicit_importance FLOAT NOT NULL DEFAULT 0.5,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ttl_expires         TIMESTAMPTZ,
    deleted_at          TIMESTAMPTZ,
    is_active           BOOLEAN NOT NULL DEFAULT true,
    pinned              BOOLEAN NOT NULL DEFAULT false,
    conflict_tag        conflict_tag NOT NULL DEFAULT 'none',
    needs_manual_review BOOLEAN NOT NULL DEFAULT false,
    dedup_checked_at    TIMESTAMPTZ,
    conflict_checked_at TIMESTAMPTZ,
    session_id          UUID REFERENCES sessions(id) ON DELETE SET NULL,
    supersedes          UUID REFERENCES memory_entries(id) ON DELETE SET NULL,
    superseded_by       UUID REFERENCES memory_entries(id) ON DELETE SET NULL,
    linked_conflict_id  UUID REFERENCES memory_entries(id) ON DELETE SET NULL,
    outcome_feedback    memory_outcome,
    ep_session_start    TIMESTAMPTZ,
    ep_is_summarized    BOOLEAN NOT NULL DEFAULT false,
    ep_summarized_at    TIMESTAMPTZ,
    proc_task_type      TEXT,
    proc_version        INTEGER DEFAULT 1,
    proc_success_count  INTEGER DEFAULT 0,
    proc_failure_count  INTEGER DEFAULT 0,
    proc_last_used      TIMESTAMPTZ,
    detail              JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memory_agent_type        ON memory_entries(agent_id, memory_type) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_scope             ON memory_entries(scope) WHERE is_active = true AND scope = 'global';
CREATE INDEX IF NOT EXISTS idx_memory_importance        ON memory_entries(importance_score DESC) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed_at  ON memory_entries(last_accessed_at DESC) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_ttl_expires       ON memory_entries(ttl_expires ASC) WHERE is_active = true AND ttl_expires IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_sync_status       ON memory_entries(sync_status) WHERE sync_status != 'synced';
CREATE INDEX IF NOT EXISTS idx_memory_proc_task_type    ON memory_entries(proc_task_type, agent_id) WHERE memory_type = 'procedural' AND is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_conflict_tag      ON memory_entries(conflict_tag) WHERE conflict_tag != 'none' AND is_active = true;
CREATE INDEX IF NOT EXISTS idx_memory_deleted_at        ON memory_entries(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_content_trgm      ON memory_entries USING GIN (content gin_trgm_ops) WHERE is_active = true;

-- ============================================================
-- TABLE: memory_access_log
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_access_log (
    id                  BIGSERIAL PRIMARY KEY,
    memory_id           UUID NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    session_id          UUID REFERENCES sessions(id) ON DELETE SET NULL,
    agent_id            UUID REFERENCES agents(id) ON DELETE SET NULL,
    accessed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    outcome_recorded    memory_outcome,
    retrieval_rank      INTEGER,
    retrieval_score     FLOAT
);
CREATE INDEX IF NOT EXISTS idx_access_log_memory_id  ON memory_access_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_log_session_id ON memory_access_log(session_id);
CREATE INDEX IF NOT EXISTS idx_access_log_accessed_at ON memory_access_log(accessed_at DESC);

-- ============================================================
-- TABLE: memory_health_reports
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_health_reports (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id            UUID REFERENCES agents(id) ON DELETE CASCADE,
    report_type         TEXT NOT NULL DEFAULT 'daily_summary',
    severity            TEXT NOT NULL DEFAULT 'info',
    details             JSONB NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_health_reports_agent_id   ON memory_health_reports(agent_id);
CREATE INDEX IF NOT EXISTS idx_health_reports_created_at ON memory_health_reports(created_at DESC);

-- ============================================================
-- TABLE: deletion_audit_log
-- ============================================================
CREATE TABLE IF NOT EXISTS deletion_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    memory_id           UUID NOT NULL,
    agent_id            UUID,
    memory_type         memory_type,
    deleted_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hard_deleted_at     TIMESTAMPTZ,
    reason              TEXT,
    snapshot            JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_deletion_audit_memory_id ON deletion_audit_log(memory_id);

-- ============================================================
-- ROW-LEVEL SECURITY
-- ============================================================
ALTER TABLE memory_entries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS agent_read_own_and_global ON memory_entries;
CREATE POLICY agent_read_own_and_global ON memory_entries
    FOR SELECT
    USING (
        agent_id = current_setting('app.current_agent_id', true)::UUID
        OR scope = 'global'
    );

DROP POLICY IF EXISTS agent_write_own ON memory_entries;
CREATE POLICY agent_write_own ON memory_entries
    FOR INSERT
    WITH CHECK (
        agent_id = current_setting('app.current_agent_id', true)::UUID
        AND scope = 'agent'
    );

DROP POLICY IF EXISTS elevated_write_global ON memory_entries;
CREATE POLICY elevated_write_global ON memory_entries
    FOR INSERT
    WITH CHECK (
        current_setting('app.agent_scope_tier', true) IN ('elevated', 'operator')
    );

DROP POLICY IF EXISTS agent_update_own ON memory_entries;
CREATE POLICY agent_update_own ON memory_entries
    FOR UPDATE
    USING (
        agent_id = current_setting('app.current_agent_id', true)::UUID
        OR current_setting('app.agent_scope_tier', true) = 'operator'
    );

-- ============================================================
-- TRIGGERS
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_agents_updated_at ON agents;
CREATE TRIGGER trg_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE FUNCTION auto_register_task_type()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.task_type_id IS NULL AND NEW.metadata->>'task_type_name' IS NOT NULL THEN
        INSERT INTO task_types (name)
        VALUES (NEW.metadata->>'task_type_name')
        ON CONFLICT (name) DO NOTHING;

        SELECT id INTO NEW.task_type_id
        FROM task_types
        WHERE name = NEW.metadata->>'task_type_name';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sessions_auto_task_type ON sessions;
CREATE TRIGGER trg_sessions_auto_task_type
    BEFORE INSERT ON sessions
    FOR EACH ROW EXECUTE FUNCTION auto_register_task_type();