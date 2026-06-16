"""
Weaviate client singleton and schema setup.
All HNSW parameters and alpha values from design spec Day 2.
"""

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

from config import settings

_client: weaviate.WeaviateClient | None = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_local(
            host=settings.weaviate_url.replace("http://", "").split(":")[0],
            port=int(settings.weaviate_url.split(":")[-1]),
            grpc_port=settings.weaviate_grpc_port,
        )
    return _client


def create_schema() -> None:
    """
    Create all three Weaviate collections with multi-tenancy enabled.
    HNSW parameters from Day 2 design spec.
    Must be called once at startup if collections do not exist.
    """
    client = get_weaviate_client()

    collections = {c.name for c in client.collections.list_all().values()}

    # ── SemanticMemory ───────────────────────────────────────────────────
    if "SemanticMemory" not in collections:
        client.collections.create(
            name="SemanticMemory",
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=128,
                ef_construction=256,
                max_connections=64,
                distance_metric=VectorDistances.COSINE,
            ),
            properties=[
                Property(name="fact", data_type=DataType.TEXT),
                Property(name="fact_type", data_type=DataType.TEXT),
                Property(name="entities", data_type=DataType.TEXT_ARRAY),
                Property(name="confidence", data_type=DataType.NUMBER),
                Property(name="agent_id", data_type=DataType.TEXT),
                Property(name="scope", data_type=DataType.TEXT),
                Property(name="importance_score", data_type=DataType.NUMBER),
                Property(name="postgres_id", data_type=DataType.UUID),
            ],
        )
        print("Created SemanticMemory collection")

    # ── ProceduralMemory ─────────────────────────────────────────────────
    if "ProceduralMemory" not in collections:
        client.collections.create(
            name="ProceduralMemory",
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=64,
                ef_construction=256,
                max_connections=64,
                distance_metric=VectorDistances.COSINE,
            ),
            properties=[
                Property(name="trigger_condition", data_type=DataType.TEXT),
                Property(name="task_type", data_type=DataType.TEXT),
                Property(name="confidence", data_type=DataType.NUMBER),
                Property(name="agent_id", data_type=DataType.TEXT),
                Property(name="postgres_id", data_type=DataType.UUID),
            ],
        )
        print("Created ProceduralMemory collection")

    # ── EpisodicMemory ───────────────────────────────────────────────────
    if "EpisodicMemory" not in collections:
        client.collections.create(
            name="EpisodicMemory",
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                ef=64,
                ef_construction=128,
                max_connections=64,
                distance_metric=VectorDistances.COSINE,
            ),
            properties=[
                Property(name="task_prompt", data_type=DataType.TEXT),
                Property(name="task_type", data_type=DataType.TEXT),
                Property(name="outcome", data_type=DataType.TEXT),
                Property(name="agent_id", data_type=DataType.TEXT),
                Property(name="session_start", data_type=DataType.DATE),
                Property(name="postgres_id", data_type=DataType.UUID),
            ],
        )
        print("Created EpisodicMemory collection")


# ── Alpha values from Day 2 benchmark ────────────────────────────────────
ALPHA = {
    "SemanticMemory": 0.7,
    "ProceduralMemory": 0.4,
    "EpisodicMemory": 0.5,
}

# ── k values per memory type (Day 2) ─────────────────────────────────────
K = {
    "SemanticMemory": 5,
    "ProceduralMemory": 1,
    "EpisodicMemory": 3,
}
