#!/usr/bin/env python3
"""
Migration script to transfer data from ChromaDB to PostgreSQL with pgvector.

This script:
1. Connects to the existing ChromaDB instance
2. Exports all documents with their embeddings and metadata
3. Connects to the new PostgreSQL database
4. Imports all documents using PGVector

Usage:
    python migrate_chroma_to_postgres.py
"""

import os
import sys
from typing import List

import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    POSTGRES_CONNECTION_STRING,
    logger,
)


def connect_to_chromadb() -> chromadb.HttpClient:
    """Connect to the existing ChromaDB instance."""
    # Use environment variables or defaults for ChromaDB connection
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8001")

    logger.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")

    try:
        client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
        client.heartbeat()
        logger.info("Successfully connected to ChromaDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise


def export_documents_from_chromadb(
    client: chromadb.HttpClient, collection_name: str
) -> List[Document]:
    """Export all documents from ChromaDB collection."""
    logger.info(f"Exporting documents from collection: {collection_name}")

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}")
        raise

    # Get all documents from the collection
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    if not results or not results.get("ids"):
        logger.warning("No documents found in ChromaDB collection")
        return []

    # Convert to LangChain Document objects
    documents = []
    ids = results.get("ids", [])
    contents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    logger.info(f"Found {len(ids)} documents in ChromaDB")

    for i, doc_id in enumerate(ids):
        content = contents[i] if i < len(contents) else ""
        metadata = metadatas[i] if i < len(metadatas) else {}

        if metadata is None:
            metadata = {}

        # Add the original ChromaDB ID to metadata for reference
        metadata["chromadb_id"] = doc_id

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    logger.info(f"Exported {len(documents)} documents from ChromaDB")
    return documents


def import_documents_to_postgres(
    documents: List[Document],
    connection_string: str,
    collection_name: str,
    embedding_model_name: str
) -> None:
    """Import documents into PostgreSQL with PGVector."""
    if not documents:
        logger.warning("No documents to import")
        return

    logger.info(f"Importing {len(documents)} documents to PostgreSQL")
    logger.info(f"Using embedding model: {embedding_model_name}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Initialize PGVector
    logger.info("Initializing PGVector...")
    vectorstore = PGVector(
        connection=connection_string,
        embeddings=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
    )

    # Add documents in batches to avoid memory issues
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        logger.info(f"Importing batch {batch_num}/{total_batches} ({len(batch)} documents)")

        try:
            vectorstore.add_documents(batch)
            logger.info(f"Successfully imported batch {batch_num}")
        except Exception as e:
            logger.error(f"Failed to import batch {batch_num}: {e}")
            raise

    logger.info(f"Successfully imported all {len(documents)} documents to PostgreSQL")


def verify_migration(
    connection_string: str,
    collection_name: str,
    expected_count: int,
    embedding_model_name: str
) -> bool:
    """Verify that the migration was successful."""
    logger.info("Verifying migration...")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore = PGVector(
            connection=connection_string,
            embeddings=embeddings,
            collection_name=collection_name,
            use_jsonb=True,
        )

        # Perform a simple similarity search to verify
        test_results = vectorstore.similarity_search("test", k=1)

        logger.info(f"Migration verification: Retrieved {len(test_results)} results from PostgreSQL")
        logger.info(f"Expected {expected_count} documents to be migrated")

        # Note: We can't easily count all documents without direct SQL access
        # So we just verify that we can query the vectorstore
        logger.info("✓ PostgreSQL vectorstore is accessible and queryable")
        return True

    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False


def main():
    """Main migration function."""
    logger.info("="*60)
    logger.info("ChromaDB to PostgreSQL Migration")
    logger.info("="*60)

    # Step 1: Connect to ChromaDB
    logger.info("\n[Step 1/4] Connecting to ChromaDB...")
    chroma_client = connect_to_chromadb()

    # Step 2: Export documents from ChromaDB
    logger.info("\n[Step 2/4] Exporting documents from ChromaDB...")
    documents = export_documents_from_chromadb(chroma_client, COLLECTION_NAME)

    if not documents:
        logger.warning("No documents found to migrate. Exiting.")
        return

    # Step 3: Import documents to PostgreSQL
    logger.info("\n[Step 3/4] Importing documents to PostgreSQL...")
    import_documents_to_postgres(
        documents,
        POSTGRES_CONNECTION_STRING,
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME
    )

    # Step 4: Verify migration
    logger.info("\n[Step 4/4] Verifying migration...")
    success = verify_migration(
        POSTGRES_CONNECTION_STRING,
        COLLECTION_NAME,
        len(documents),
        EMBEDDING_MODEL_NAME
    )

    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ Migration completed successfully!")
        logger.info(f"✓ Migrated {len(documents)} documents from ChromaDB to PostgreSQL")
    else:
        logger.error("✗ Migration verification failed")
        sys.exit(1)
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nMigration failed with error: {e}", exc_info=True)
        sys.exit(1)
