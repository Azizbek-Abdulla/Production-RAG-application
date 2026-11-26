"""
Qdrant vector store: collection setup, upserting embedded chunks, and search.

Usage:
    from database.vector_store import get_qdrant_client, ensure_collection, upsert_chunks
    client = get_qdrant_client()
    ensure_collection(client)
    upsert_chunks(client, chunks, embeddings)
"""
from qdrant_client import QdrantClient
from rag.embeddings import get_embedder, embed_query
from qdrant_client.http import models as qmodels
from app.config import settings
from utils.log import get_logger
import uuid

logger = get_logger(__name__)

embedder = get_embedder()
def get_qdrant_client() -> QdrantClient:
    """
    Creates a Qdrant client from settings. Call this once and reuse this client 
    rather than creating a new one per request.
    """
    logger.info("connecting to Qdrant", url=settings.qdrant_url)

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

    return client

def ensure_collection(client: QdrantClient) -> None:
    """
    Creates a collection if it doesn't already exist. Safe to call on every app
    startup -- it is a no-op if the collection is already there.
    """
    collection_name = settings.qdrant_collection
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.info("collection already exists", collection=collection_name) 
        return

    logger.info(
        "Creating collection",
        collection=collection_name,
        dim=settings.embedding_dim
    )

    client.create_collection(
        collection_name=collection_name, 
        vectors_config={"dense": 
                        qmodels.VectorParams(
                            size=settings.embedding_dim,
                            distance=qmodels.Distance.COSINE
                            )}, 
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()}
    )

def upsert_chunks(
        client: QdrantClient, 
        chunks: list[dict], 
        embeddings: tuple[list[list[float]], list[qmodels.SparseVector]]
) -> None:
    """
    Writes chunks and their embeddings into Qdrant vector store as points.
    chunks: list of Chunk objects from rag/chunker.py
    embeddings: list of vectors, same length and order as chunks.
    """
    if len(chunks) != len(embeddings):
        logger.error(
            "Chunks/embedding size mismatch occured",
            num_chunks=len(chunks), 
            num_embeddings=len(embeddings)
        )

        raise ValueError("chunks and embeddings must the same length")

    ids = [str(uuid.uuid4) for _ in range(1, len(chunks)+1)]
    dense_vecs, sparse_vecs = embeddings
    if not chunks:
        return 
    payloads = [
        c if isinstance(c, dict) else {
            "text": c["text"],
            "source": c["source"],
            "page": c["page"],
            "chunk_id": getattr(c, "chunk_id", None)
        }
        for c in chunks
    ]

    points = qmodels.Batch(ids=ids, 
                           vectors={
                               "dense": dense_vecs,
                               "sparse": sparse_vecs
                           },
                           payloads=payloads)
    
    client.upsert(
        collection_name=settings.qdrant_collection, 
        points=points
    )

    logger.info(
        "Upserted chunks into Qdrant", 
        collection=settings.qdrant_collection, 
        num_points=len(points)
    )

def search(
    client: QdrantClient, 
    user_query_raw: str,
    top_k: int = 20) -> list[dict]: 
    """
    Runs a dense vectors search and returns matches as plain dicts
    (text, source, page, score) for every downstream use
    """
    logger.info("Running vector search", top_k=top_k)
    os_query_dense, os_query_sparse = embed_query(embedder=embedder, query=user_query_raw)
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query_vector=qmodels.Prefetch(
            prefetch=[
                qmodels.Prefetch(
                    query=os_query_dense,
                    using="dense",
                    limit=20
                ),
                qmodels.Prefetch(
                    query=os_query_sparse,
                    using="sparse",
                    limit=20
                )
            ],
            query = qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                           limit=10,
                           with_payload=True
                           ))

    matches = [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "score": r.score
        }
        for r in results.points
    ]

    logger.info("Search complete", num_matches=len(matches))
    return matches