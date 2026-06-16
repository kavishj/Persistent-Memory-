import os
import pytest
from adapter.base import MemoryAdapter, RetrievedContext, TaskResult

class LiveAdapter(MemoryAdapter):
    def get_agent_id(self) -> str:
        return "test-adapter-agent"

    def get_api_key(self) -> str:
        return os.environ.get("MEMORY_ENGINE_API_KEY", "test-key-standard")

    def get_engine_url(self) -> str:
        return os.environ.get("MEMORY_ENGINE_URL", "http://localhost:8000")

    def serialize_session_log(self, raw: object) -> str:
        import json
        return json.dumps(raw)

@pytest.mark.asyncio
async def test_live_adapter_integration():
    import httpx
    url = os.environ.get("MEMORY_ENGINE_URL", "http://localhost:8000")
    try:
        resp = httpx.get(f"{url}/ping", timeout=2.0)
        resp.raise_for_status()
    except Exception:
        pytest.skip("Live memory engine not reachable")

    adapter = LiveAdapter()
    
    ctx = await adapter.pre_task("Integration test prompt")
    assert isinstance(ctx, RetrievedContext)
    assert ctx.session_id is not None
    
    result = TaskResult(
        session_id=ctx.session_id,
        output="Result output",
        outcome="success",
        task_type="integration",
        raw_session_log={"step": "integration_test"}
    )
    
    queued = await adapter.post_task(result)
    assert isinstance(queued, bool)
