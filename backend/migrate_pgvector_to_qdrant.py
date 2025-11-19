#!/usr/bin/env python3
"""
Migration script to transfer documents from PostgreSQL/PGVector to Qdrant.

Usage:
    python migrate_pgvector_to_qdrant.py [--collection COLLECTION_NAME] [--batch-size BATCH_SIZE]

Example:
    python migrate_pgvector_to_qdrant.py --collection rag_documents --batch-size 100
"""

import argparse
import sys
from typing import Optional

import psycopg
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    POSTGRES_LIBPQ_CONNECTION,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    logger,
)


def get_pgvector_documents(collection_name: str) -> list[dict]:
    """
    Retrieve all documents from PostgreSQL/PGVector.

    Returns list of dicts with: id, document (content), cmetadata, embedding
    """
    documents = []

    with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
        with conn.cursor() as cur:
            # Query to get all documents with their embeddings
            cur.execute(
                """
                SELECT e.id, e.document, e.cmetadata, e.embedding
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                ORDER BY e.id
                """,
                (collection_name,)
            )

            results = cur.fetchall()

            for row in results:
                doc_id, document, metadata, embedding = row
                documents.append({
                    "id": str(doc_id),
                    "content": document or "",
                    "metadata": metadata or {},
                    "embedding": embedding,
                })

    return documents


def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int,
) -> None:
    """Create a Qdrant collection if it doesn't exist."""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        logger.info(f"Collection '{collection_name}' already exists in Qdrant")
        # Optionally delete and recreate
        response = input(f"Collection '{collection_name}' exists. Delete and recreate? (y/N): ")
        if response.lower() == 'y':
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        else:
            logger.info("Keeping existing collection. Documents will be added to it.")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Created Qdrant collection '{collection_name}' with dimension {embedding_dim}")


def migrate_to_qdrant(
    documents: list[dict],
    collection_name: str,
    batch_size: int = 100,
) -> int:
    """
    Migrate documents to Qdrant.

    Returns the number of documents migrated.
    """
    if not documents:
        logger.warning("No documents to migrate")
        return 0

    # Initialize Qdrant client
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )

    # Get embedding dimension from first document
    first_embedding = documents[0].get("embedding")
    if first_embedding is None:
        logger.error("Documents don't have embeddings. Cannot migrate.")
        return 0

    embedding_dim = len(first_embedding)
    logger.info(f"Detected embedding dimension: {embedding_dim}")

    # Create collection
    create_qdrant_collection(client, collection_name, embedding_dim)

    # Prepare points for Qdrant
    points = []
    for i, doc in enumerate(documents):
        embedding = doc.get("embedding")
        if embedding is None:
            logger.warning(f"Skipping document {doc['id']} - no embedding")
            continue

        # Create point with payload containing document content and metadata
        point = PointStruct(
            id=i,  # Use integer IDs for Qdrant
            vector=embedding,
            payload={
                "page_content": doc["content"],
                "metadata": doc["metadata"],
                "original_id": doc["id"],  # Keep original PGVector ID for reference
            },
        )
        points.append(point)

    # Upload in batches
    total_uploaded = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        total_uploaded += len(batch)
        logger.info(f"Uploaded {total_uploaded}/{len(points)} documents")

    return total_uploaded


def verify_migration(collection_name: str) -> dict:
    """Verify the migration by checking document counts and running a test query."""
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )

    # Get collection info
    info = client.get_collection(collection_name)

    return {
        "collection": collection_name,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": info.status.value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate documents from PostgreSQL/PGVector to Qdrant"
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection name to migrate (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for uploading to Qdrant (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be migrated without actually migrating",
    )

    args = parser.parse_args()

    logger.info(f"Starting migration from PGVector to Qdrant")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    # Get documents from PGVector
    logger.info("Fetching documents from PostgreSQL/PGVector...")
    documents = get_pgvector_documents(args.collection)

    if not documents:
        logger.warning(f"No documents found in collection '{args.collection}'")
        sys.exit(0)

    logger.info(f"Found {len(documents)} documents to migrate")

    if args.dry_run:
        logger.info("Dry run - not migrating")
        logger.info(f"Would migrate {len(documents)} documents to Qdrant collection '{args.collection}'")
        sys.exit(0)

    # Migrate to Qdrant
    logger.info("Migrating documents to Qdrant...")
    migrated_count = migrate_to_qdrant(
        documents=documents,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )

    logger.info(f"Successfully migrated {migrated_count} documents")

    # Verify migration
    logger.info("Verifying migration...")
    verification = verify_migration(args.collection)
    logger.info(f"Verification results: {verification}")

    if verification["points_count"] == migrated_count:
        logger.info("Migration completed successfully!")
    else:
        logger.warning(
            f"Document count mismatch: migrated {migrated_count}, "
            f"found {verification['points_count']} in Qdrant"
        )


if __name__ == "__main__":
    main()
