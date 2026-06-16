import pytest
import uuid
from fastapi.testclient import TestClient
from api.main import app, resolve_api_key, AuthContext, get_db

client = TestClient(app)

def test_expire_operator_requires_scope():
    # 1. No auth header (FastAPI validation fails)
    resp = client.request("DELETE", "/memory/expire", json={"memory_ids": ["123"], "reason": "test"})
    assert resp.status_code == 422

    # 1b. Invalid auth key
    # (Without dependency override, it will try database which might fail, but let's test)
    # Actually, let's keep dependency override for testing scope-tier standard which expects 403
    app.dependency_overrides[resolve_api_key] = lambda: AuthContext(agent_id="test", scope_tier="standard")
    resp = client.request("DELETE", "/memory/expire", json={"memory_ids": ["123"], "reason": "test"}, headers={"X-API-Key": "test"})
    assert resp.status_code == 403
    assert "Operator scope required" in resp.json()["detail"]

    # 3. Operator scope, DB mock
    app.dependency_overrides[resolve_api_key] = lambda: AuthContext(agent_id="test", scope_tier="operator")
    
    class MockDB:
        def execute(self, *args, **kwargs):
            class MockResult:
                def fetchall(self):
                    return []
            return MockResult()
    
    app.dependency_overrides[get_db] = lambda: MockDB()
    
    mem_id = str(uuid.uuid4())
    resp = client.request("DELETE", "/memory/expire", json={"memory_ids": [mem_id], "reason": "test"})
    assert resp.status_code == 404
    assert "No active memories found" in resp.json()["detail"]

    app.dependency_overrides.clear()

def test_expire_operator_success():
    app.dependency_overrides[resolve_api_key] = lambda: AuthContext(agent_id="test-operator", scope_tier="operator")
    
    class MockRow:
        def __init__(self, mem_id):
            self.id = mem_id
            self.agent_id = "test-agent"
            self.memory_type = "semantic"
            self.content = "test content"
            self.importance_score = 0.5
            self.created_at = "2026-01-01"

    mem_id = str(uuid.uuid4())

    class MockDB:
        def execute(self, query, params=None):
            class MockResult:
                def fetchall(self):
                    return [MockRow(mem_id)]
            return MockResult()
        def commit(self):
            pass

    app.dependency_overrides[get_db] = lambda: MockDB()
    
    resp = client.request("DELETE", "/memory/expire", json={"memory_ids": [mem_id], "reason": "operator test manual expire"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["expired"] == 1
    assert len(data["audit_ids"]) == 1

    app.dependency_overrides.clear()
