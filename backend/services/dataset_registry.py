"""
Dataset registry service for managing multiple datasets with different embedders.
Handles CRUD operations for dataset configurations.
"""

from datetime import datetime, timezone
from typing import List, Optional

import psycopg
from fastapi import HTTPException
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from config import POSTGRES_LIBPQ_CONNECTION
from database_models import Dataset, get_session_maker
from schemas import (
    DatasetCreateRequest,
    DatasetInfo,
    DatasetListResponse,
    DatasetMetadata,
    EmbedderConfig,
)


class DatasetRegistryService:
    """Service for managing dataset configurations and metadata."""

    def __init__(self):
        self.SessionLocal = get_session_maker()

    def create_dataset(self, request: DatasetCreateRequest) -> DatasetInfo:
        """Create a new dataset with the specified configuration."""
        session = self.SessionLocal()
        try:
            # Check if dataset with this name already exists
            existing = session.query(Dataset).filter(Dataset.name == request.name).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset with name '{request.name}' already exists"
                )

            # Generate collection name
            collection_name = f"dataset_{request.name}"

            # Create dataset record
            dataset = Dataset(
                name=request.name,
                description=request.description,
                collection_name=collection_name,
                embedder_model_name=request.embedder_config.model_name,
                embedder_model_type=request.embedder_config.model_type,
                embedder_dimensions=request.embedder_config.dimensions,
                embedder_model_kwargs=request.embedder_config.model_kwargs or {},
            )

            session.add(dataset)
            session.commit()
            session.refresh(dataset)

            # Create the PGVector collection in PostgreSQL
            self._initialize_collection(collection_name)

            return self._dataset_to_info(dataset)

        except HTTPException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create dataset: {str(e)}"
            ) from e
        finally:
            session.close()

    def list_datasets(self) -> DatasetListResponse:
        """List all datasets with their metadata."""
        session = self.SessionLocal()
        try:
            datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).all()

            # Update document and chunk counts for each dataset
            for dataset in datasets:
                self._update_dataset_counts(session, dataset)

            session.commit()

            dataset_infos = [self._dataset_to_info(d) for d in datasets]
            return DatasetListResponse(count=len(dataset_infos), datasets=dataset_infos)

        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list datasets: {str(e)}"
            ) from e
        finally:
            session.close()

    def get_dataset(self, name: str) -> DatasetInfo:
        """Get a specific dataset by name."""
        session = self.SessionLocal()
        try:
            dataset = session.query(Dataset).filter(Dataset.name == name).first()
            if not dataset:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{name}' not found"
                )

            # Update counts before returning
            self._update_dataset_counts(session, dataset)
            session.commit()

            return self._dataset_to_info(dataset)

        except HTTPException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get dataset: {str(e)}"
            ) from e
        finally:
            session.close()

    def delete_dataset(self, name: str) -> dict:
        """Delete a dataset and its associated collection."""
        session = self.SessionLocal()
        try:
            dataset = session.query(Dataset).filter(Dataset.name == name).first()
            if not dataset:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{name}' not found"
                )

            collection_name = dataset.collection_name

            # Delete dataset record
            session.delete(dataset)
            session.commit()

            # Delete the collection from PostgreSQL
            try:
                self._delete_collection(collection_name)
            except Exception as e:
                # Log the error but don't fail the operation
                print(f"Warning: Failed to delete collection '{collection_name}': {e}")

            return {"message": f"Dataset '{name}' deleted successfully"}

        except HTTPException:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete dataset: {str(e)}"
            ) from e
        finally:
            session.close()

    def update_dataset_counts(self, dataset_name: str) -> None:
        """Update the document and chunk counts for a dataset."""
        session = self.SessionLocal()
        try:
            dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
            if dataset:
                self._update_dataset_counts(session, dataset)
                session.commit()
        finally:
            session.close()

    def _update_dataset_counts(self, session: Session, dataset: Dataset) -> None:
        """Update document and chunk counts for a dataset by querying the collection."""
        try:
            with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
                with conn.cursor() as cur:
                    # Count total chunks in this collection
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        """,
                        (dataset.collection_name,)
                    )
                    chunk_count = cur.fetchone()[0] or 0

                    # Count unique documents (based on original_filename in metadata)
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT cmetadata->>'original_filename')
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        AND cmetadata ? 'original_filename'
                        """,
                        (dataset.collection_name,)
                    )
                    doc_count = cur.fetchone()[0] or 0

            dataset.chunk_count = chunk_count
            dataset.document_count = doc_count
            dataset.updated_at = datetime.now(timezone.utc)

        except Exception as e:
            print(f"Warning: Failed to update counts for dataset '{dataset.name}': {e}")
            # Don't raise, just keep existing counts

    def _initialize_collection(self, collection_name: str) -> None:
        """Initialize a PGVector collection in PostgreSQL."""
        try:
            import json
            import uuid
            with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
                with conn.cursor() as cur:
                    # Insert into langchain_pg_collection if not exists
                    # Generate a UUID and convert empty dict to JSON string for psycopg
                    collection_uuid = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
                        VALUES (%s, %s, %s::jsonb)
                        ON CONFLICT (name) DO NOTHING
                        """,
                        (collection_uuid, collection_name, json.dumps({}))
                    )
                conn.commit()
        except Exception as e:
            raise Exception(f"Failed to initialize collection: {str(e)}") from e

    def _delete_collection(self, collection_name: str) -> None:
        """Delete a PGVector collection and all its embeddings from PostgreSQL."""
        try:
            with psycopg.connect(POSTGRES_LIBPQ_CONNECTION) as conn:
                with conn.cursor() as cur:
                    # First delete all embeddings for this collection
                    cur.execute(
                        """
                        DELETE FROM langchain_pg_embedding
                        WHERE collection_id IN (
                            SELECT uuid FROM langchain_pg_collection WHERE name = %s
                        )
                        """,
                        (collection_name,)
                    )

                    # Then delete the collection itself
                    cur.execute(
                        "DELETE FROM langchain_pg_collection WHERE name = %s",
                        (collection_name,)
                    )
                conn.commit()
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}") from e

    def _dataset_to_info(self, dataset: Dataset) -> DatasetInfo:
        """Convert a Dataset model to DatasetInfo schema."""
        embedder_config = EmbedderConfig(
            model_name=dataset.embedder_model_name,
            model_type=dataset.embedder_model_type,
            dimensions=dataset.embedder_dimensions,
            model_kwargs=dataset.embedder_model_kwargs or {}
        )

        metadata = DatasetMetadata(
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            document_count=dataset.document_count,
            chunk_count=dataset.chunk_count
        )

        return DatasetInfo(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            collection_name=dataset.collection_name,
            embedder_config=embedder_config,
            metadata=metadata
        )
