"""
Evaluation router - API endpoints for RAG evaluation.

Provides endpoints for:
- Creating and managing evaluation datasets (Q&A pairs)
- Running evaluations with configurable parameters
- Retrieving evaluation results and metrics
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio
import csv
import io
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from schemas.evaluation import (
    AnswerCorrectnessDetail,
    ChunkingConfig,
    ClaimVerificationDetail,
    CreateEvalDatasetRequest,
    EvalConfig,
    EvalDatasetResponse,
    EvalMetrics,
    EvalResult,
    EvalResultScores,
    EXPERIMENT_PRESETS,
    FaithfulnessDetail,
    GeneratedPair,
    GenerateQARequest,
    GenerateQAResponse,
    GenerateSampleRequest,
    GenerateSampleResponse,
    RetrievedChunk,
    RunEvaluationRequest,
    RunEvaluationResponse,
)
from services.qa_generator import QAGeneratorService
from services.embedding_service import embedding_service
from services.file_loader import file_loader_service
from services.chunking import ChunkingService
from services.raw_dataset import raw_dataset_service
from database_models import RawDataset, RawFile
from schemas import RawDatasetCreate, SourceTypeEnum
from database_models import get_db_session
from services.evaluation_metrics import RAGEvaluationMetrics
from rag_components import get_rag_context_prefix, format_docs
from config import VLLM_BASE_URL, VLLM_MODEL
import httpx

logger = logging.getLogger(__name__)

# Cache for current vLLM model
_current_vllm_model: Optional[str] = None


async def _get_current_vllm_model() -> str:
    """Fetch the currently loaded model from vLLM API.

    Handles dynamic model switching - the env var doesn't update
    when models are switched, so we query the API directly.
    """
    global _current_vllm_model
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    _current_vllm_model = models[0].get("id")
                    return _current_vllm_model
    except Exception as e:
        logger.warning(f"Could not fetch current vLLM model: {e}")

    return _current_vllm_model or VLLM_MODEL


# Directory for storing evaluation datasets
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Directory for storing evaluation runs (results)
EVAL_RUNS_DIR = Path("data/evaluation_runs")
EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(
    prefix="/api/evaluation",
    tags=["evaluation"],
)


def _get_dataset_path(dataset_id: str) -> Path:
    """Get the file path for a dataset."""
    return EVAL_DATASETS_DIR / f"{dataset_id}.json"


def _load_dataset(dataset_id: str) -> dict:
    """Load a dataset from disk."""
    path = _get_dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    with open(path) as f:
        return json.load(f)


def _save_dataset(dataset_id: str, data: dict):
    """Save a dataset to disk."""
    path = _get_dataset_path(dataset_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@router.get("/datasets", response_model=list[EvalDatasetResponse])
async def list_eval_datasets():
    """List all evaluation datasets."""
    datasets = []
    for path in EVAL_DATASETS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                datasets.append(
                    EvalDatasetResponse(
                        id=data["id"],
                        name=data["name"],
                        pair_count=len(data["pairs"]),
                        created_at=data["created_at"],
                        source_collection=data.get("source_collection"),  # May be None for old datasets
                    )
                )
        except Exception as e:
            logger.error(f"Error loading dataset {path}: {e}")
    return datasets


@router.post("/datasets", response_model=EvalDatasetResponse)
async def create_eval_dataset(request: CreateEvalDatasetRequest):
    """Create a new evaluation dataset with Q&A pairs."""
    dataset_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()

    data = {
        "id": dataset_id,
        "name": request.name,
        "pairs": [p.model_dump() for p in request.pairs],
        "created_at": created_at,
    }

    _save_dataset(dataset_id, data)
    logger.info(f"Created evaluation dataset '{request.name}' with {len(request.pairs)} pairs")

    return EvalDatasetResponse(
        id=dataset_id,
        name=request.name,
        pair_count=len(request.pairs),
        created_at=created_at,
    )


@router.get("/datasets/{dataset_id}")
async def get_eval_dataset(dataset_id: str):
    """Get a specific evaluation dataset with all Q&A pairs."""
    return _load_dataset(dataset_id)


@router.delete("/datasets/{dataset_id}")
async def delete_eval_dataset(dataset_id: str):
    """Delete an evaluation dataset."""
    path = _get_dataset_path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    path.unlink()
    return {"status": "deleted", "id": dataset_id}


# Pre-configured HuggingFace datasets for one-click import
HUGGINGFACE_DATASETS = {
    "microsoft/wiki_qa": {
        "name": "Microsoft Wiki QA",
        "description": "Q&A pairs from Wikipedia, ~3k questions",
        "split": "train",
        "question_field": "question",
        "answer_field": "answer",
        "filter_fn": lambda row: row.get("label") == 1,  # Only positive examples
    },
    "squad": {
        "name": "SQuAD v1.1",
        "description": "Stanford Question Answering Dataset, ~87k questions",
        "split": "train",
        "question_field": "question",
        "answer_field": "answers",  # SQuAD has answers as dict with 'text' list
        "answer_extractor": lambda row: row["answers"]["text"][0] if row["answers"]["text"] else "",
    },
    "trivia_qa": {
        "name": "TriviaQA",
        "description": "Trivia questions with evidence documents",
        "config": "rc",
        "split": "train",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": lambda row: row["answer"]["value"] if row["answer"] else "",
    },
    "hotpot_qa": {
        "name": "HotpotQA",
        "description": "Multi-hop reasoning questions",
        "config": "fullwiki",
        "split": "train",
        "question_field": "question",
        "answer_field": "answer",
    },
}


@router.get("/import/datasets")
async def list_importable_datasets():
    """List available HuggingFace datasets for import."""
    return [
        {
            "id": dataset_id,
            "name": config["name"],
            "description": config["description"],
        }
        for dataset_id, config in HUGGINGFACE_DATASETS.items()
    ]


@router.post("/import/huggingface")
async def import_huggingface_dataset(
    dataset_id: str,
    limit: int = 500,
    custom_name: Optional[str] = None,
):
    """
    Import a dataset from HuggingFace.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., 'squad', 'microsoft/wiki_qa')
        limit: Maximum number of rows to import
        custom_name: Optional custom name for the dataset
    """
    if dataset_id not in HUGGINGFACE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{dataset_id}' not in pre-configured list. Available: {list(HUGGINGFACE_DATASETS.keys())}"
        )

    config = HUGGINGFACE_DATASETS[dataset_id]

    try:
        from datasets import load_dataset

        # Load dataset from HuggingFace
        load_args = [dataset_id]
        load_kwargs = {"split": config["split"]}
        if "config" in config:
            load_args.append(config["config"])

        logger.info(f"Loading dataset {dataset_id} from HuggingFace...")
        hf_dataset = load_dataset(*load_args, **load_kwargs)

        # Extract Q&A pairs
        pairs = []
        filter_fn = config.get("filter_fn")
        answer_extractor = config.get("answer_extractor")

        for i, row in enumerate(hf_dataset):
            if limit and len(pairs) >= limit:
                break

            # Apply filter if defined
            if filter_fn and not filter_fn(row):
                continue

            question = row[config["question_field"]]

            # Extract answer based on dataset format
            if answer_extractor:
                answer = answer_extractor(row)
            else:
                answer = row[config["answer_field"]]

            # Skip empty answers
            if not answer or not question:
                continue

            pairs.append({
                "query": question,
                "ground_truth": answer,
            })

        if not pairs:
            raise HTTPException(status_code=400, detail="No valid Q&A pairs found in dataset")

        # Create and save dataset
        new_dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()
        dataset_name = custom_name or f"{config['name']} ({len(pairs)} pairs)"

        data = {
            "id": new_dataset_id,
            "name": dataset_name,
            "pairs": pairs,
            "created_at": created_at,
            "source": f"huggingface:{dataset_id}",
            "import_limit": limit,
        }

        _save_dataset(new_dataset_id, data)
        logger.info(f"Imported {len(pairs)} pairs from {dataset_id}")

        return EvalDatasetResponse(
            id=new_dataset_id,
            name=dataset_name,
            pair_count=len(pairs),
            created_at=created_at,
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="'datasets' library not installed. Run: pip install datasets"
        )
    except Exception as e:
        logger.error(f"Failed to import dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# Directory for bundled PDF datasets
BUNDLED_PDFS_DIR = Path("data/eval_pdfs")
BUNDLED_PDFS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-configured PDF datasets for one-click import
# local_file: if set, load from bundled file instead of URL
PDF_DATASETS = [
    {
        "id": "constitutia-romaniei",
        "name": "Constitutia Romaniei",
        "description": "Constitutia Romaniei (2003) - document juridic fundamental",
        "url": "https://www.ccr.ro/wp-content/uploads/2020/03/Constitutia-2003.pdf",
        "local_file": "constitutia-romaniei.pdf",
        "language": "ro",
        "pages": 67,
    },
    {
        "id": "carte-bucate-sanda-marin",
        "name": "Carte de Bucate - Sanda Marin",
        "description": "Carte de bucate traditionala romaneasca",
        "url": "https://apiardeal.ro/biblioteca/carti/GASTRONOMIE/Carte_de_bucate_-_Sanda_Marin_-_255_pag.pdf",
        "local_file": "carte-bucate-sanda-marin.pdf",
        "language": "ro",
        "pages": 255,
    },
    {
        "id": "istoria-romanilor",
        "name": "Elemente de Istorie a Romaniei",
        "description": "Manual de istorie a romanilor",
        "url": "https://www.sociouman-usamvb.ro/documents/Elemente_de_istorie_a_Romaniei.pdf",
        "local_file": "istoria-romanilor.pdf",
        "language": "ro",
        "pages": 128,
    },
]


@router.get("/import/pdf-datasets")
async def list_pdf_datasets():
    """List available pre-configured PDF datasets for import."""
    return PDF_DATASETS


class PDFImportStatus(BaseModel):
    """Status of a PDF import job."""
    status: str  # pending, downloading, extracting, chunking, generating, completed, failed
    progress: int  # 0-100
    message: str
    pairs_generated: int = 0
    dataset_id: Optional[str] = None
    raw_dataset_id: Optional[int] = None  # ID of created raw dataset (if also_create_raw_dataset=True)
    error: Optional[str] = None


# In-memory storage for import job status (in production, use Redis or DB)
_pdf_import_jobs: dict[str, PDFImportStatus] = {}


@router.get("/import/pdf-status/{job_id}")
async def get_pdf_import_status(job_id: str):
    """Get the status of a PDF import job."""
    if job_id not in _pdf_import_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _pdf_import_jobs[job_id]


async def _process_pdf_import(
    job_id: str,
    pdf_content: bytes,
    filename: str,
    dataset_name: str,
    max_pairs: int,
    pairs_per_chunk: int,
    use_vllm: bool,
    also_create_raw_dataset: bool = False,
):
    """Background task to process PDF and generate Q&A pairs."""
    raw_dataset_id = None

    try:
        # Create raw dataset first if requested
        if also_create_raw_dataset:
            try:
                _pdf_import_jobs[job_id] = PDFImportStatus(
                    status="creating_raw",
                    progress=5,
                    message="Creating raw dataset...",
                )

                with get_db_session() as db:
                    # Check if dataset with this name already exists
                    existing = db.query(RawDataset).filter(RawDataset.name == dataset_name).first()
                    if existing:
                        raw_dataset_id = existing.id
                        logger.info(f"Raw dataset '{dataset_name}' already exists (id={raw_dataset_id})")
                    else:
                        # Create new raw dataset
                        raw_dataset = RawDataset(
                            name=dataset_name,
                            description=f"Imported from PDF: {filename}",
                            source_type="upload",
                        )
                        db.add(raw_dataset)
                        db.flush()
                        raw_dataset_id = raw_dataset.id

                        # Add PDF file to the dataset
                        content_hash = hashlib.sha256(pdf_content).hexdigest()
                        raw_file = RawFile(
                            raw_dataset_id=raw_dataset_id,
                            filename=filename,
                            file_type="pdf",
                            mime_type="application/pdf",
                            file_content=pdf_content,
                            size_bytes=len(pdf_content),
                            content_hash=content_hash,
                        )
                        db.add(raw_file)

                        # Update dataset file count and size
                        raw_dataset.total_file_count = 1
                        raw_dataset.total_size_bytes = len(pdf_content)
                        db.commit()

                        logger.info(f"Created raw dataset '{dataset_name}' (id={raw_dataset_id}) with PDF")
            except Exception as e:
                logger.warning(f"Failed to create raw dataset: {e}")
                # Continue with Q&A generation even if raw dataset creation fails
        # Update status: extracting
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="extracting",
            progress=10,
            message=f"Extracting text from {filename}...",
        )

        # Load PDF
        documents = file_loader_service.load_file(
            content=pdf_content,
            filename=filename,
            file_type="pdf",
        )

        if not documents:
            raise ValueError("No text could be extracted from PDF")

        logger.info(f"Extracted {len(documents)} pages from PDF")

        # Update status: chunking
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="chunking",
            progress=20,
            message=f"Chunking {len(documents)} pages...",
        )

        # Chunk the documents
        chunking_service = ChunkingService()
        chunked_docs = chunking_service.chunk_documents(
            documents=documents,
            method="recursive",
            chunk_size=1500,  # Larger chunks for Q&A generation
            chunk_overlap=200,
        )

        logger.info(f"Created {len(chunked_docs)} chunks")

        # Limit chunks based on max_pairs
        max_chunks = min(len(chunked_docs), max_pairs // pairs_per_chunk + 1)
        selected_chunks = chunked_docs[:max_chunks]

        # Update status: generating
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="generating",
            progress=30,
            message=f"Generating Q&A pairs from {len(selected_chunks)} chunks...",
        )

        # Initialize QA generator
        qa_generator = QAGeneratorService(use_vllm=use_vllm)

        # Generate Q&A pairs
        all_pairs = []
        batch_size = 3
        total_chunks = len(selected_chunks)

        for i in range(0, total_chunks, batch_size):
            batch = selected_chunks[i:i + batch_size]

            # Process batch concurrently
            tasks = [
                qa_generator.generate_pairs_for_chunk(
                    chunk_content=doc.page_content,
                    source_info=f"{filename}, Page {doc.metadata.get('page', 'N/A')}",
                    pairs_count=pairs_per_chunk,
                )
                for doc in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, list):
                    all_pairs.extend(result)

            # Update progress
            progress = 30 + int((i + len(batch)) / total_chunks * 60)
            _pdf_import_jobs[job_id] = PDFImportStatus(
                status="generating",
                progress=progress,
                message=f"Generated {len(all_pairs)} pairs from {min(i + batch_size, total_chunks)}/{total_chunks} chunks...",
                pairs_generated=len(all_pairs),
            )

            # Stop if we have enough pairs
            if len(all_pairs) >= max_pairs:
                all_pairs = all_pairs[:max_pairs]
                break

            # Small delay between batches
            await asyncio.sleep(0.5)

        if not all_pairs:
            raise ValueError("No Q&A pairs could be generated from the PDF")

        # Save dataset
        new_dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        data = {
            "id": new_dataset_id,
            "name": dataset_name,
            "pairs": all_pairs,
            "created_at": created_at,
            "source": f"pdf:{filename}",
            "source_pages": len(documents),
            "chunks_processed": len(selected_chunks),
        }

        _save_dataset(new_dataset_id, data)

        # Update status: completed
        message = f"Successfully generated {len(all_pairs)} Q&A pairs"
        if raw_dataset_id:
            message += f" and created raw dataset"

        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="completed",
            progress=100,
            message=message,
            pairs_generated=len(all_pairs),
            dataset_id=new_dataset_id,
            raw_dataset_id=raw_dataset_id,
        )

        logger.info(f"PDF import completed: {len(all_pairs)} pairs saved to dataset {new_dataset_id}, raw_dataset_id={raw_dataset_id}")

    except Exception as e:
        logger.error(f"PDF import failed: {e}", exc_info=True)
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="failed",
            progress=0,
            message="Import failed",
            error=str(e),
        )


@router.post("/import/pdf-local")
async def import_pdf_local(
    background_tasks: BackgroundTasks,
    dataset_id: str,
    max_pairs: int = 100,
    pairs_per_chunk: int = 2,
    use_vllm: bool = True,
    custom_name: Optional[str] = None,
    also_create_raw_dataset: bool = False,
):
    """
    Import a bundled PDF from local storage and generate Q&A pairs.

    Args:
        dataset_id: ID of the pre-configured PDF dataset
        max_pairs: Maximum number of Q&A pairs to generate
        pairs_per_chunk: Number of Q&A pairs per chunk
        use_vllm: Whether to use local vLLM (True) or OpenRouter (False)

    Returns:
        Job ID for tracking progress
    """
    # Find the dataset config
    dataset_config = next((d for d in PDF_DATASETS if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    local_file = dataset_config.get("local_file")
    if not local_file:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} does not have a bundled file. Use PDF upload instead."
        )

    local_path = BUNDLED_PDFS_DIR / local_file
    if not local_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundled PDF not found: {local_file}. Please ensure the file exists in data/eval_pdfs/"
        )

    job_id = str(uuid.uuid4())[:8]

    # Initialize job status
    _pdf_import_jobs[job_id] = PDFImportStatus(
        status="loading",
        progress=5,
        message=f"Loading {local_file}...",
    )

    try:
        # Read local PDF
        pdf_content = local_path.read_bytes()
        logger.info(f"Loaded {len(pdf_content)} bytes from {local_path}")

        # Start background processing
        background_tasks.add_task(
            _process_pdf_import,
            job_id=job_id,
            pdf_content=pdf_content,
            filename=local_file,
            dataset_name=custom_name or dataset_config["name"],
            max_pairs=max_pairs,
            pairs_per_chunk=pairs_per_chunk,
            use_vllm=use_vllm,
            also_create_raw_dataset=also_create_raw_dataset,
        )

        return {"job_id": job_id, "status": "started"}

    except Exception as e:
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="failed",
            progress=0,
            message="Load failed",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to load PDF: {e}")


@router.post("/import/pdf-url")
async def import_pdf_from_url(
    background_tasks: BackgroundTasks,
    url: str,
    dataset_name: str,
    max_pairs: int = 100,
    pairs_per_chunk: int = 2,
    use_vllm: bool = True,
    also_create_raw_dataset: bool = False,
):
    """
    Import a PDF from a URL and generate Q&A pairs.

    Args:
        url: URL to the PDF file
        dataset_name: Name for the evaluation dataset
        max_pairs: Maximum number of Q&A pairs to generate
        pairs_per_chunk: Number of Q&A pairs per chunk
        use_vllm: Whether to use local vLLM (True) or OpenRouter (False)

    Returns:
        Job ID for tracking progress
    """
    job_id = str(uuid.uuid4())[:8]

    # Initialize job status
    _pdf_import_jobs[job_id] = PDFImportStatus(
        status="downloading",
        progress=0,
        message=f"Downloading PDF from {url}...",
    )

    try:
        # Download PDF with browser-like headers to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9,ro;q=0.8",
            "Connection": "keep-alive",
        }
        # Try with SSL verification first, then without if it fails
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
        except Exception as ssl_err:
            logger.warning(f"SSL error, retrying without verification: {ssl_err}")
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, verify=False) as client:
                response = await client.get(url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF: HTTP {response.status_code} - {response.text[:200] if response.text else 'No response body'}"
            )
        pdf_content = response.content

        # Verify we got a PDF
        if not pdf_content or len(pdf_content) < 100:
            raise HTTPException(status_code=400, detail="Downloaded file is empty or too small")
        if not pdf_content[:4] == b'%PDF':
            raise HTTPException(status_code=400, detail="Downloaded file is not a valid PDF")

        # Extract filename from URL
        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename = f"{dataset_name}.pdf"

        logger.info(f"Downloaded {len(pdf_content)} bytes from {url}")

        # Start background processing
        background_tasks.add_task(
            _process_pdf_import,
            job_id=job_id,
            pdf_content=pdf_content,
            filename=filename,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            pairs_per_chunk=pairs_per_chunk,
            use_vllm=use_vllm,
            also_create_raw_dataset=also_create_raw_dataset,
        )

        return {"job_id": job_id, "status": "started"}

    except httpx.RequestError as e:
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="failed",
            progress=0,
            message="Download failed",
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


@router.post("/import/pdf-upload")
async def import_pdf_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    max_pairs: int = Form(100),
    pairs_per_chunk: int = Form(2),
    use_vllm: bool = Form(True),
    also_create_raw_dataset: bool = Form(False),
):
    """
    Upload a PDF file and generate Q&A pairs.

    Args:
        file: The PDF file to upload
        dataset_name: Name for the evaluation dataset
        max_pairs: Maximum number of Q&A pairs to generate
        pairs_per_chunk: Number of Q&A pairs per chunk
        use_vllm: Whether to use local vLLM (True) or OpenRouter (False)

    Returns:
        Job ID for tracking progress
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    job_id = str(uuid.uuid4())[:8]

    # Initialize job status
    _pdf_import_jobs[job_id] = PDFImportStatus(
        status="uploading",
        progress=5,
        message=f"Processing {file.filename}...",
    )

    try:
        # Read file content
        pdf_content = await file.read()

        logger.info(f"Received PDF upload: {file.filename} ({len(pdf_content)} bytes)")

        # Start background processing
        background_tasks.add_task(
            _process_pdf_import,
            job_id=job_id,
            pdf_content=pdf_content,
            filename=file.filename,
            dataset_name=dataset_name,
            max_pairs=max_pairs,
            pairs_per_chunk=pairs_per_chunk,
            use_vllm=use_vllm,
            also_create_raw_dataset=also_create_raw_dataset,
        )

        return {"job_id": job_id, "status": "started"}

    except Exception as e:
        _pdf_import_jobs[job_id] = PDFImportStatus(
            status="failed",
            progress=0,
            message="Upload failed",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")


@router.post("/import/pdf-as-raw")
async def import_pdf_as_raw(
    dataset_id: str,
    custom_name: Optional[str] = None,
):
    """
    Import a bundled PDF as a raw dataset (no Q&A generation).

    This creates a raw dataset that can then be:
    1. Preprocessed (chunking, embeddings)
    2. Used to generate Q&A pairs from the processed content

    Args:
        dataset_id: ID of the pre-configured PDF dataset
        custom_name: Optional custom name for the dataset

    Returns:
        Created raw dataset info
    """
    # Find the dataset config
    dataset_config = next((d for d in PDF_DATASETS if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    local_file = dataset_config.get("local_file")
    if not local_file:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} does not have a bundled file."
        )

    local_path = BUNDLED_PDFS_DIR / local_file
    if not local_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundled PDF not found: {local_file}. Please ensure the file exists in data/eval_pdfs/"
        )

    dataset_name = custom_name or dataset_config["name"]

    try:
        # Read PDF content
        pdf_content = local_path.read_bytes()
        content_hash = hashlib.sha256(pdf_content).hexdigest()

        with get_db_session() as db:
            # Check if dataset with this name already exists
            existing = db.query(RawDataset).filter(RawDataset.name == dataset_name).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset '{dataset_name}' already exists. Use a different name."
                )

            # Create raw dataset
            raw_dataset = RawDataset(
                name=dataset_name,
                description=f"Imported from PDF: {local_file}",
                source_type="upload",
            )
            db.add(raw_dataset)
            db.flush()

            # Add PDF file to the dataset
            raw_file = RawFile(
                raw_dataset_id=raw_dataset.id,
                filename=local_file,
                file_type="pdf",
                mime_type="application/pdf",
                file_content=pdf_content,
                size_bytes=len(pdf_content),
                content_hash=content_hash,
            )
            db.add(raw_file)

            # Update dataset stats
            raw_dataset.total_file_count = 1
            raw_dataset.total_size_bytes = len(pdf_content)
            db.commit()

            logger.info(f"Created raw dataset '{dataset_name}' (id={raw_dataset.id}) from PDF")

            return {
                "success": True,
                "raw_dataset_id": raw_dataset.id,
                "name": dataset_name,
                "file_count": 1,
                "size_bytes": len(pdf_content),
                "message": f"PDF imported as raw dataset. Go to Data Management to preprocess it."
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import PDF as raw dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to import PDF: {e}")


@router.post("/import/pdf-upload-as-raw")
async def import_pdf_upload_as_raw(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
):
    """
    Upload a PDF file as a raw dataset (no Q&A generation).

    Args:
        file: The PDF file to upload
        dataset_name: Name for the raw dataset

    Returns:
        Created raw dataset info
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_content = await file.read()
        content_hash = hashlib.sha256(pdf_content).hexdigest()

        with get_db_session() as db:
            # Check if dataset with this name already exists
            existing = db.query(RawDataset).filter(RawDataset.name == dataset_name).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset '{dataset_name}' already exists. Use a different name."
                )

            # Create raw dataset
            raw_dataset = RawDataset(
                name=dataset_name,
                description=f"Uploaded PDF: {file.filename}",
                source_type="upload",
            )
            db.add(raw_dataset)
            db.flush()

            # Add PDF file
            raw_file = RawFile(
                raw_dataset_id=raw_dataset.id,
                filename=file.filename,
                file_type="pdf",
                mime_type="application/pdf",
                file_content=pdf_content,
                size_bytes=len(pdf_content),
                content_hash=content_hash,
            )
            db.add(raw_file)

            # Update dataset stats
            raw_dataset.total_file_count = 1
            raw_dataset.total_size_bytes = len(pdf_content)
            db.commit()

            logger.info(f"Uploaded PDF as raw dataset '{dataset_name}' (id={raw_dataset.id})")

            return {
                "success": True,
                "raw_dataset_id": raw_dataset.id,
                "name": dataset_name,
                "file_count": 1,
                "size_bytes": len(pdf_content),
                "message": f"PDF uploaded as raw dataset. Go to Data Management to preprocess it."
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload PDF as raw dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {e}")


class GenerateQARequest(BaseModel):
    """Request to generate Q&A pairs from a processed dataset."""
    processed_dataset_id: int
    dataset_name: str
    max_pairs: int = 50
    pairs_per_chunk: int = 2


class GenerateQAStatus(BaseModel):
    """Status of a Q&A generation job."""
    status: str  # pending, fetching, generating, completed, failed
    progress: int  # 0-100
    message: str
    pairs_generated: int = 0
    eval_dataset_id: Optional[str] = None
    error: Optional[str] = None


# In-memory storage for Q&A generation job status
_qa_generation_jobs: dict[str, GenerateQAStatus] = {}


@router.get("/generate-qa/status/{job_id}")
async def get_qa_generation_status(job_id: str):
    """Get the status of a Q&A generation job."""
    if job_id not in _qa_generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _qa_generation_jobs[job_id]


async def _process_qa_generation(
    job_id: str,
    processed_dataset_id: int,
    dataset_name: str,
    max_pairs: int,
    pairs_per_chunk: int,
):
    """Background task to generate Q&A pairs from a processed dataset."""
    from services.processed_dataset import processed_dataset_service
    from database_models import ProcessedDataset

    try:
        _qa_generation_jobs[job_id] = GenerateQAStatus(
            status="fetching",
            progress=5,
            message="Fetching chunks from processed dataset...",
        )

        # Get all chunks from the processed dataset
        all_chunks = []
        page = 1
        limit = 100

        with get_db_session() as db:
            # Get dataset info
            dataset = db.query(ProcessedDataset).filter(ProcessedDataset.id == processed_dataset_id).first()
            if not dataset:
                raise ValueError(f"Processed dataset {processed_dataset_id} not found")

            # Fetch chunks in pages
            while True:
                result = processed_dataset_service.get_chunks(db, processed_dataset_id, page=page, limit=limit)
                chunks = result.get("chunks", [])
                if not chunks:
                    break
                all_chunks.extend(chunks)
                if len(all_chunks) >= result.get("total", 0):
                    break
                page += 1

        if not all_chunks:
            raise ValueError("No chunks found in the processed dataset")

        logger.info(f"Fetched {len(all_chunks)} chunks from processed dataset {processed_dataset_id}")

        _qa_generation_jobs[job_id] = GenerateQAStatus(
            status="generating",
            progress=20,
            message=f"Generating Q&A pairs from {len(all_chunks)} chunks...",
        )

        # Limit chunks based on max_pairs
        max_chunks = min(len(all_chunks), max_pairs // pairs_per_chunk + 1)
        selected_chunks = all_chunks[:max_chunks]

        # Generate Q&A pairs
        qa_generator = QAGeneratorService()
        all_pairs = []

        batch_size = 5
        total_chunks = len(selected_chunks)

        for i in range(0, total_chunks, batch_size):
            batch = selected_chunks[i:i + batch_size]

            for chunk in batch:
                chunk_text = chunk.get("content") or chunk.get("text", "")
                if not chunk_text or len(chunk_text) < 100:
                    continue

                result = await qa_generator.generate_qa_pairs(
                    text=chunk_text,
                    num_pairs=pairs_per_chunk,
                    use_vllm=True,
                )
                if result:
                    all_pairs.extend(result)

            # Update progress
            progress = 20 + int((i + len(batch)) / total_chunks * 70)
            _qa_generation_jobs[job_id] = GenerateQAStatus(
                status="generating",
                progress=progress,
                message=f"Generated {len(all_pairs)} pairs from {min(i + batch_size, total_chunks)}/{total_chunks} chunks...",
                pairs_generated=len(all_pairs),
            )

            # Stop if we have enough pairs
            if len(all_pairs) >= max_pairs:
                all_pairs = all_pairs[:max_pairs]
                break

            await asyncio.sleep(0.5)

        if not all_pairs:
            raise ValueError("No Q&A pairs could be generated from the chunks")

        # Save evaluation dataset
        new_dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        data = {
            "id": new_dataset_id,
            "name": dataset_name,
            "pairs": all_pairs,
            "created_at": created_at,
            "source": f"processed_dataset:{processed_dataset_id}",
            "chunks_processed": len(selected_chunks),
        }

        _save_dataset(new_dataset_id, data)

        _qa_generation_jobs[job_id] = GenerateQAStatus(
            status="completed",
            progress=100,
            message=f"Successfully generated {len(all_pairs)} Q&A pairs",
            pairs_generated=len(all_pairs),
            eval_dataset_id=new_dataset_id,
        )

        logger.info(f"Q&A generation completed: {len(all_pairs)} pairs saved to dataset {new_dataset_id}")

    except Exception as e:
        logger.error(f"Q&A generation failed: {e}", exc_info=True)
        _qa_generation_jobs[job_id] = GenerateQAStatus(
            status="failed",
            progress=0,
            message="Generation failed",
            error=str(e),
        )


@router.post("/generate-qa")
async def generate_qa_from_processed(
    background_tasks: BackgroundTasks,
    request: GenerateQARequest,
):
    """
    Generate Q&A pairs from a processed dataset.

    This takes chunks from a processed dataset and uses the LLM to generate
    question-answer pairs for evaluation.

    Args:
        processed_dataset_id: ID of the processed dataset
        dataset_name: Name for the evaluation dataset
        max_pairs: Maximum number of Q&A pairs to generate
        pairs_per_chunk: Number of Q&A pairs per chunk

    Returns:
        Job ID for tracking progress
    """
    job_id = str(uuid.uuid4())[:8]

    _qa_generation_jobs[job_id] = GenerateQAStatus(
        status="starting",
        progress=0,
        message="Starting Q&A generation...",
    )

    background_tasks.add_task(
        _process_qa_generation,
        job_id=job_id,
        processed_dataset_id=request.processed_dataset_id,
        dataset_name=request.dataset_name,
        max_pairs=request.max_pairs,
        pairs_per_chunk=request.pairs_per_chunk,
    )

    return {"job_id": job_id, "status": "started"}


@router.post("/run", response_model=RunEvaluationResponse)
async def run_evaluation(request: RunEvaluationRequest):
    """
    Run an evaluation using the specified dataset and configuration.

    Supports two modes:
    1. Legacy mode: Use existing collection_name (already chunked/embedded)
    2. New mode: Use preprocessed_dataset_id + chunking/embedder config (on-demand processing)

    Uses RAGAS-style metrics:
    - Answer Correctness: F1 factual + semantic similarity
    - Faithfulness: LLM-verified claim support from context
    - Response Relevancy: Embedding-based query-answer similarity
    - Context Precision: Mean precision@k for retrieval ranking

    This will:
    1. Load the evaluation dataset (or use test query if no dataset)
    2. For each Q&A pair:
       - Run the RAG pipeline with the query
       - Record the predicted answer, retrieved chunks, and latency
       - Calculate RAGAS-style metrics
    3. Compute aggregate metrics
    """
    # Determine effective configuration (apply preset if specified)
    effective_chunking = request.chunking
    effective_embedder = request.embedder
    effective_top_k = request.top_k
    effective_colbert = request.use_colbert
    effective_temperature = request.temperature
    preset_name = None

    if request.preset and request.preset in EXPERIMENT_PRESETS:
        preset = EXPERIMENT_PRESETS[request.preset]
        preset_name = preset.name
        # Use preset values unless explicitly overridden
        if not request.chunking:
            effective_chunking = preset.chunking
        if request.embedder == "sentence-transformers/all-MiniLM-L6-v2":  # Default value
            effective_embedder = preset.embedder
        if request.top_k == 5:  # Default value
            effective_top_k = preset.top_k
        if not request.use_colbert:  # Default is False
            effective_colbert = preset.reranker == "colbert"
        if request.temperature == 0.1:  # Default value
            effective_temperature = preset.temperature

        logger.info(f"Using preset '{preset_name}': {preset.description}")

    # Determine collection name
    collection_name = request.collection_name
    was_cached = False

    # New mode: Use preprocessed_dataset_id with on-demand chunking/embedding
    # Only use new flow if dataset actually has preprocessed documents
    if request.preprocessed_dataset_id and effective_chunking:
        from database_models import PreprocessedDocument, ProcessedDataset
        from sqlalchemy import func

        # Get database session
        db_gen = get_db_session()
        db = next(db_gen)

        try:
            # Check if dataset has preprocessed documents (new flow)
            preprocessed_count = db.query(func.count(PreprocessedDocument.id)).filter(
                PreprocessedDocument.processed_dataset_id == request.preprocessed_dataset_id
            ).scalar()

            if preprocessed_count > 0:
                # New flow: dataset has preprocessed documents, use on-demand embedding
                logger.info(
                    f"New flow: Creating/getting collection for dataset {request.preprocessed_dataset_id} "
                    f"with chunking {effective_chunking.method}/{effective_chunking.chunk_size} "
                    f"and embedder {effective_embedder}"
                )

                # Get or create collection using EmbeddingService
                collection_name, was_cached = await embedding_service.get_or_create_collection(
                    db=db,
                    preprocessed_dataset_id=request.preprocessed_dataset_id,
                    chunking_config=effective_chunking,
                    embedder_model=effective_embedder,
                )

                logger.info(
                    f"Using collection '{collection_name}' (cached={was_cached})"
                )
            else:
                # Legacy flow: dataset was processed with direct indexing
                # Get collection name from ProcessedDataset table
                processed_ds = db.query(ProcessedDataset).filter(
                    ProcessedDataset.id == request.preprocessed_dataset_id
                ).first()

                if processed_ds and processed_ds.collection_name:
                    collection_name = processed_ds.collection_name
                    logger.info(
                        f"Dataset {request.preprocessed_dataset_id} has no preprocessed documents, "
                        f"using legacy collection '{collection_name}'"
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset {request.preprocessed_dataset_id} has no preprocessed documents and no collection. "
                               "Please re-process the dataset."
                    )
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    # Get pairs from dataset or use test query
    if request.eval_dataset_id:
        dataset = _load_dataset(request.eval_dataset_id)
        pairs = dataset["pairs"]
        dataset_name = dataset["name"]
    else:
        # Use test query or default
        test_query = request.test_query or "What information is available in the documents?"
        pairs = [{"query": test_query, "ground_truth": "Based on retrieved documents."}]
        dataset_name = "Quick Test"

    logger.info(
        f"Starting evaluation: dataset='{dataset_name}', "
        f"pairs={len(pairs)}, collection={collection_name}, "
        f"rag={request.use_rag}, colbert={effective_colbert}"
    )

    # Initialize RAGAS-style metrics evaluator
    metrics_evaluator = RAGEvaluationMetrics()

    results = []
    total_relevancy = 0.0
    total_faithfulness = 0.0
    total_correctness = 0.0
    total_context_precision = 0.0
    total_latency = 0.0

    for i, pair in enumerate(pairs):
        query = pair["query"]
        ground_truth = pair["ground_truth"]

        logger.info(f"Evaluating pair {i+1}/{len(pairs)}: {query[:50]}...")

        start_time = time.time()

        try:
            # Get RAG context and generate answer
            if request.use_rag:
                # Get RAG context prefix
                rag_prefix = await get_rag_context_prefix(
                    query,
                    collection_name=collection_name,
                    use_colbert=effective_colbert,
                    embedder=effective_embedder,
                )
                prompt_content = rag_prefix if rag_prefix else query
            else:
                prompt_content = query

            # Call vLLM to generate answer
            predicted_answer, retrieved_chunks = await _generate_answer(
                prompt_content,
                query,
                effective_temperature,
                request.use_rag,
            )

            generation_latency = time.time() - start_time

            # Extract chunk contents for metrics
            chunk_contents = [c.content for c in retrieved_chunks] if retrieved_chunks else []

            # Calculate Jaccard similarity (original metric - kept for reference)
            jaccard_score = _calculate_jaccard(predicted_answer, ground_truth)

            # Calculate RAGAS-style metrics
            logger.info(f"  Computing metrics for pair {i+1}...")

            # Answer Correctness (F1 + semantic)
            correctness_result = await metrics_evaluator.answer_correctness(
                predicted_answer, ground_truth
            )

            # Faithfulness (LLM claim verification)
            faithfulness_result = await metrics_evaluator.faithfulness(
                predicted_answer, chunk_contents
            )

            # Response Relevancy (embedding similarity)
            relevancy = metrics_evaluator.response_relevancy(query, predicted_answer)

            # Context Precision (retrieval ranking quality)
            precision_result = await metrics_evaluator.context_precision(
                query, chunk_contents, ground_truth
            )

            total_latency += generation_latency
            total_relevancy += relevancy
            total_faithfulness += faithfulness_result.score
            total_correctness += correctness_result.score
            total_context_precision += precision_result.score

            # Build detailed score breakdown
            faithfulness_detail = FaithfulnessDetail(
                total_claims=faithfulness_result.total_claims,
                supported_claims=faithfulness_result.supported_claims,
                claims=[
                    ClaimVerificationDetail(
                        claim=c.claim,
                        supported=c.supported,
                        evidence=c.evidence,
                    )
                    for c in faithfulness_result.claim_details
                ],
            )

            correctness_detail = AnswerCorrectnessDetail(
                factual_score=correctness_result.factual_score,
                semantic_score=correctness_result.semantic_score,
                true_positives=correctness_result.true_positives,
                false_positives=correctness_result.false_positives,
                false_negatives=correctness_result.false_negatives,
            )

            results.append(
                EvalResult(
                    query=query,
                    ground_truth=ground_truth,
                    predicted_answer=predicted_answer,
                    retrieved_chunks=retrieved_chunks,
                    score=jaccard_score,  # Keep Jaccard as main score for backwards compat
                    scores=EvalResultScores(
                        jaccard=jaccard_score,
                        relevancy=relevancy,
                        faithfulness=faithfulness_result.score,
                        answer_correctness=correctness_result.score,
                        context_precision=precision_result.score,
                        faithfulness_detail=faithfulness_detail,
                        answer_correctness_detail=correctness_detail,
                    ),
                    latency=generation_latency,
                )
            )

        except Exception as e:
            logger.error(f"Error evaluating pair {i+1}: {e}", exc_info=True)
            results.append(
                EvalResult(
                    query=query,
                    ground_truth=ground_truth,
                    predicted_answer=f"Error: {str(e)}",
                    retrieved_chunks=None,
                    score=0.0,
                    scores=None,
                    latency=time.time() - start_time,
                )
            )

    # Calculate aggregate metrics
    n = len([r for r in results if r.scores is not None])  # Only count successful evaluations
    metrics = EvalMetrics(
        answer_relevancy=total_relevancy / n if n > 0 else 0.0,
        faithfulness=total_faithfulness / n if n > 0 else 0.0,
        context_precision=total_context_precision / n if n > 0 else 0.0,
        answer_correctness=total_correctness / n if n > 0 else 0.0,
        avg_latency=total_latency / len(results) if results else 0.0,
    )

    logger.info(
        f"Evaluation complete: "
        f"answer_correctness={metrics.answer_correctness:.2f}, "
        f"faithfulness={metrics.faithfulness:.2f}, "
        f"relevancy={metrics.answer_relevancy:.2f}, "
        f"context_precision={metrics.context_precision:.2f}, "
        f"avg_latency={metrics.avg_latency:.2f}s"
    )

    # Build response
    response = RunEvaluationResponse(
        eval_dataset_id=request.eval_dataset_id,
        results=results,
        metrics=metrics,
        config=EvalConfig(
            collection=collection_name,
            use_rag=request.use_rag,
            use_colbert=effective_colbert,
            top_k=effective_top_k,
            temperature=effective_temperature,
            embedder=effective_embedder,
            llm_model=VLLM_MODEL,
            preprocessed_dataset_id=request.preprocessed_dataset_id,
            chunking=effective_chunking,
            preset_name=preset_name,
        ),
    )

    # Save evaluation run to disk for persistence
    run_id = str(uuid.uuid4())[:8]
    run_data = {
        "id": run_id,
        "name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "eval_dataset_id": request.eval_dataset_id,
        "config": response.config.model_dump(),
        "metrics": response.metrics.model_dump(),
        "results": [r.model_dump() for r in response.results],
    }
    run_path = EVAL_RUNS_DIR / f"{run_id}.json"
    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    logger.info(f"Saved evaluation run to {run_path}")

    return response


async def _generate_answer(
    prompt_content: str,
    original_query: str,
    temperature: float,
    use_rag: bool,
) -> tuple[str, Optional[list[RetrievedChunk]]]:
    """Generate an answer using vLLM."""
    # Get current model from vLLM API (handles dynamic model switching)
    current_model = await _get_current_vllm_model()

    messages = [{"role": "user", "content": prompt_content}]

    payload = {
        "model": current_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise Exception(f"vLLM error: {response.status_code}")

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        # Extract retrieved chunks from the prompt if RAG was used
        retrieved_chunks = None
        if use_rag and "Context:" in prompt_content:
            # Parse context from the prompt
            try:
                context_start = prompt_content.find("Context:") + len("Context:")
                question_start = prompt_content.find("Question:")
                if question_start > context_start:
                    context_text = prompt_content[context_start:question_start].strip()
                    # Split into chunks (simplified - assumes double newline separation)
                    chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
                    retrieved_chunks = [
                        RetrievedChunk(content=chunk[:500], source="retrieved")
                        for chunk in chunks[:5]
                    ]
            except Exception:
                pass

        return answer, retrieved_chunks


def _calculate_jaccard(predicted: str, ground_truth: str) -> float:
    """
    Calculate Jaccard similarity between predicted answer and ground truth.
    Simple word overlap metric (kept for backwards compatibility/reference).
    """
    pred_words = set(predicted.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not pred_words or not truth_words:
        return 0.0

    intersection = pred_words & truth_words
    union = pred_words | truth_words

    return len(intersection) / len(union) if union else 0.0


# --- Evaluation Runs (History & Export) ---

@router.get("/runs")
async def list_evaluation_runs():
    """List all past evaluation runs."""
    runs = []
    for path in sorted(EVAL_RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                config = data.get("config", {})
                # Backfill llm_model for existing runs that don't have it
                if "llm_model" not in config:
                    config["llm_model"] = VLLM_MODEL
                runs.append({
                    "id": data["id"],
                    "name": data.get("name", "Unknown"),
                    "created_at": data["created_at"],
                    "pair_count": len(data.get("results", [])),
                    "metrics": data.get("metrics", {}),
                    "config": config,
                })
        except Exception as e:
            logger.error(f"Error loading run {path}: {e}")
    return runs


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """Get a specific evaluation run with all results."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(path) as f:
        data = json.load(f)
    # Backfill llm_model for existing runs that don't have it
    if "config" in data and "llm_model" not in data["config"]:
        data["config"]["llm_model"] = VLLM_MODEL
    return data


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    """Delete an evaluation run."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    path.unlink()
    return {"status": "deleted", "id": run_id}


@router.get("/runs/{run_id}/csv")
async def export_evaluation_run_csv(run_id: str):
    """Export an evaluation run as CSV."""
    path = EVAL_RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    with open(path) as f:
        data = json.load(f)

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "Query",
        "Predicted Answer",
        "Ground Truth",
        "Jaccard",
        "Answer Correctness",
        "Faithfulness",
        "Context Precision",
        "Relevancy",
        "Latency (s)",
    ])

    # Data rows
    for result in data.get("results", []):
        scores = result.get("scores") or {}
        writer.writerow([
            result.get("query", ""),
            result.get("predicted_answer", ""),
            result.get("ground_truth", ""),
            f"{(scores.get('jaccard') or result.get('score', 0)) * 100:.1f}%",
            f"{(scores.get('answer_correctness') or 0) * 100:.1f}%",
            f"{(scores.get('faithfulness') or 0) * 100:.1f}%",
            f"{(scores.get('context_precision') or 0) * 100:.1f}%",
            f"{(scores.get('relevancy') or 0) * 100:.1f}%",
            f"{result.get('latency', 0):.2f}",
        ])

    # Add summary row
    metrics = data.get("metrics", {})
    writer.writerow([])
    writer.writerow(["SUMMARY METRICS"])
    writer.writerow(["Answer Correctness", f"{metrics.get('answer_correctness', 0) * 100:.1f}%"])
    writer.writerow(["Faithfulness", f"{metrics.get('faithfulness', 0) * 100:.1f}%"])
    writer.writerow(["Context Precision", f"{metrics.get('context_precision', 0) * 100:.1f}%"])
    writer.writerow(["Answer Relevancy", f"{metrics.get('answer_relevancy', 0) * 100:.1f}%"])
    writer.writerow(["Avg Latency", f"{metrics.get('avg_latency', 0):.2f}s"])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=evaluation_{run_id}.csv"}
    )


# --- Q&A Generation Endpoints ---

@router.post("/datasets/generate", response_model=GenerateQAResponse)
async def generate_qa_dataset(request: GenerateQARequest):
    """
    Generate Q&A pairs from a processed dataset using LLM.

    Supports two LLM providers:
    - OpenRouter (default): Cloud-based, uses GPT-4o-mini or other models
    - Local vLLM: Uses your local vLLM instance

    This will:
    1. Retrieve chunks from the processed dataset
    2. Use the selected LLM to generate Q&A pairs from each chunk
    3. Save the generated pairs as an evaluation dataset
    """
    try:
        generator = QAGeneratorService(
            model=request.model,
            use_vllm=request.use_vllm,
            temperature=request.temperature,
        )

        result = await generator.generate_pairs_for_dataset(
            processed_dataset_id=request.processed_dataset_id,
            name=request.name,
            pairs_per_chunk=request.pairs_per_chunk,
            max_chunks=request.max_chunks,
            seed=request.seed,
        )

        return GenerateQAResponse(
            id=result["id"],
            name=result["name"],
            pair_count=result["pair_count"],
            chunks_processed=result["chunks_processed"],
            created_at=result["created_at"],
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating Q&A dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/datasets/generate/sample", response_model=GenerateSampleResponse)
async def generate_sample_pairs(request: GenerateSampleRequest):
    """
    Generate a small sample of Q&A pairs for preview.

    Use this to preview what generated Q&A pairs will look like
    before running full generation.
    """
    try:
        generator = QAGeneratorService()

        pairs = await generator.generate_sample_pairs(
            processed_dataset_id=request.processed_dataset_id,
            sample_size=request.sample_size,
        )

        return GenerateSampleResponse(
            pairs=[GeneratedPair(**pair) for pair in pairs],
            chunks_sampled=request.sample_size,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating sample pairs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sample generation failed: {str(e)}")


# --- New Batch, Dry-Run, and Import Endpoints ---

@router.post("/batch")
async def run_batch_evaluation(configs: list[dict], dataset_id: Optional[str] = None):
    """
    Run batch evaluations with multiple configurations.

    Useful for comparing different embedder/reranker combinations.
    """
    from services.eval_runner import BatchEvalRunner, EvalConfig

    try:
        eval_configs = [EvalConfig.from_dict(c) for c in configs]

        batch_runner = BatchEvalRunner()
        results = await batch_runner.run_batch(eval_configs, dataset_id, parallel=True)

        return {
            "batch_id": str(uuid.uuid4())[:8],
            "run_ids": [r.run_id for r in results],
            "config_hashes": [r.config.config_hash() for r in results],
        }
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/dry-run")
async def dry_run_evaluation(
    collection_name: str,
    eval_dataset_id: Optional[str] = None,
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_colbert: bool = False,
    top_k: int = 5,
    temperature: float = 0.1,
    use_rag: bool = True,
):
    """
    Validate evaluation configuration without running inference.

    Returns:
    - config_hash: Unique hash for this configuration
    - would_create_new_run: Whether this config has been run before
    - validation_errors: Any configuration issues
    """
    from services.eval_runner import EvalConfig, EvalRunner

    config = EvalConfig(
        embedder_model=embedder,
        reranker_strategy="colbert" if use_colbert else "none",
        top_k=top_k,
        temperature=temperature,
        dataset_id=eval_dataset_id,
        collection_name=collection_name,
        use_rag=use_rag,
    )

    runner = EvalRunner()
    result = runner.dry_run(config, eval_dataset_id)

    return {
        "config_hash": result["config_hash"],
        "would_create_new_run": "existing_run_id" not in result,
        "existing_run_id": result.get("existing_run_id"),
        "valid": result["valid"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "dataset_name": result.get("dataset_name"),
        "pair_count": result.get("pair_count"),
    }


@router.get("/metrics")
async def list_available_metrics():
    """
    List all available evaluation metrics.

    Returns metadata about each metric including:
    - name: Metric identifier
    - description: What the metric measures
    - requires_context: Whether it needs retrieved chunks
    - requires_reference: Whether it needs ground truth
    """
    from services.metrics.registry import get_all_metric_info

    return get_all_metric_info()


@router.post("/runs/{run_id}/rescore")
async def rescore_evaluation_run(run_id: str, metrics: Optional[list[str]] = None):
    """
    Re-score an evaluation run with new or different metrics.

    Useful when new metrics are added or you want to recompute
    specific metrics without re-running inference.
    """
    from services.eval_scorer import EvalScorer

    try:
        scorer = EvalScorer()
        scored_result = await scorer.rescore_run(run_id, metrics)
        scorer.save_scored_run(scored_result)

        return {
            "run_id": scored_result.run_id,
            "metrics": scored_result.metrics.to_dict(),
            "scored_at": scored_result.scored_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Re-scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-scoring failed: {str(e)}")


@router.get("/datasets/import/formats")
async def list_import_formats():
    """List available dataset import formats."""
    from services.dataset_importers import list_importers

    return list_importers()


@router.post("/datasets/import")
async def import_dataset(
    format: str,
    name: str,
    file: bytes,
    max_pairs: Optional[int] = None,
):
    """
    Import an evaluation dataset from a standard format.

    Supported formats:
    - squad: Stanford Question Answering Dataset
    - natural_questions: Google Natural Questions
    - msmarco: Microsoft MARCO

    The file should be uploaded as raw bytes.
    """
    from services.dataset_importers import get_importer
    from services.dataset_validator import validate_qa_pairs
    from io import BytesIO

    try:
        importer = get_importer(format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = importer.import_from_stream(BytesIO(file), max_pairs)

        if not result.success or not result.pairs:
            raise HTTPException(
                status_code=400,
                detail=f"Import failed: {result.errors[0] if result.errors else 'No pairs found'}",
            )

        # Validate imported pairs
        validation = validate_qa_pairs([p.to_dict() for p in result.pairs])

        if not validation.valid:
            logger.warning(f"Validation issues: {validation.errors}")

        # Convert to standard format and save
        dataset_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        pairs = [
            {
                "query": p.question,
                "ground_truth": p.expected_answer,
                "alternative_answers": p.alternative_answers,
                "answer_type": p.answer_type,
                "difficulty": p.difficulty,
                "is_answerable": p.is_answerable,
                "metadata": p.metadata,
            }
            for p in result.pairs
        ]

        data = {
            "id": dataset_id,
            "name": name,
            "pairs": pairs,
            "created_at": created_at,
            "source_format": result.source_format,
            "import_stats": {
                "total_processed": result.total_processed,
                "skipped": result.skipped,
                "imported": len(result.pairs),
            },
        }

        _save_dataset(dataset_id, data)

        return {
            "id": dataset_id,
            "name": name,
            "pair_count": len(result.pairs),
            "source_format": result.source_format,
            "created_at": created_at,
            "validation": validation.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str):
    """
    Validate an evaluation dataset.

    Checks for:
    - Empty questions/answers
    - Duplicate questions
    - Format consistency
    """
    from services.dataset_validator import validate_qa_pairs

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        result = validate_qa_pairs(pairs)

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/runs/compare")
async def compare_evaluation_runs(
    run_id_a: str,
    run_id_b: str,
    metrics: Optional[list[str]] = None,
    alpha: float = 0.05,
):
    """
    Statistically compare two evaluation runs.

    Computes:
    - Paired t-test / Wilcoxon signed-rank test
    - Cohen's d effect size
    - Significance determination with multiple testing correction

    Args:
        run_id_a: First run ID
        run_id_b: Second run ID
        metrics: Metrics to compare (default: all available)
        alpha: Significance level (default: 0.05)

    Returns:
        Statistical comparison results with p-values and effect sizes
    """
    from services.statistics import (
        compare_runs,
        bootstrap_ci,
        benjamini_hochberg_correction,
        summarize_comparison,
    )

    # Load both runs
    path_a = EVAL_RUNS_DIR / f"{run_id_a}.json"
    path_b = EVAL_RUNS_DIR / f"{run_id_b}.json"

    if not path_a.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id_a} not found")
    if not path_b.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id_b} not found")

    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    results_a = data_a.get("results", [])
    results_b = data_b.get("results", [])

    if len(results_a) != len(results_b):
        raise HTTPException(
            status_code=400,
            detail=f"Runs have different number of results ({len(results_a)} vs {len(results_b)}). "
                   "Cannot perform paired comparison."
        )

    # Default metrics to compare
    if metrics is None:
        metrics = ["answer_correctness", "faithfulness", "context_precision", "relevancy"]

    # Extract scores for each metric
    comparisons = []
    for metric_name in metrics:
        scores_a = []
        scores_b = []

        for r_a, r_b in zip(results_a, results_b):
            # Get scores from the results
            score_a = r_a.get("scores", {}).get(metric_name, 0.0) if r_a.get("scores") else 0.0
            score_b = r_b.get("scores", {}).get(metric_name, 0.0) if r_b.get("scores") else 0.0
            scores_a.append(score_a)
            scores_b.append(score_b)

        if scores_a and scores_b:
            comparison = compare_runs(scores_a, scores_b, metric_name, alpha=alpha)
            comparisons.append(comparison)

    # Apply multiple testing correction
    if comparisons:
        p_values = [c.p_value for c in comparisons if c.p_value is not None]
        if p_values:
            significant, adjusted_p = benjamini_hochberg_correction(p_values, alpha)
            # Update significance based on correction
            for i, comp in enumerate(comparisons):
                if i < len(significant):
                    comp.is_significant = significant[i]

    # Generate summary
    summary = summarize_comparison(comparisons, alpha)

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "comparisons": [c.to_dict() for c in comparisons],
        "alpha": alpha,
        "summary": summary,
        "n_samples": len(results_a),
    }


# --- Dataset Contamination & Split Endpoints ---


@router.post("/datasets/{dataset_id}/check-contamination")
async def check_dataset_contamination(
    dataset_id: str,
    training_dataset_id: Optional[str] = None,
    check_ngram: bool = True,
    check_semantic: bool = False,
):
    """
    Check for contamination between evaluation dataset and training data.

    Args:
        dataset_id: Evaluation dataset to check
        training_dataset_id: Training dataset to check against (optional)
        check_ngram: Enable n-gram overlap detection
        check_semantic: Enable semantic similarity detection (slower)

    Returns:
        Contamination detection results
    """
    from services.contamination_checker import ContaminationChecker

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        checker = ContaminationChecker()

        if training_dataset_id:
            # Check against another dataset
            training_dataset = _load_dataset(training_dataset_id)
            training_pairs = training_dataset.get("pairs", [])
            # Extract text content from training pairs
            training_corpus = []
            for p in training_pairs:
                q = p.get("question", p.get("query", ""))
                a = p.get("expected_answer", p.get("ground_truth", ""))
                training_corpus.extend([q, a])

            result = checker.check_contamination(
                pairs,
                training_corpus,
                check_exact=True,
                check_ngram=check_ngram,
                check_semantic=check_semantic,
            )
        else:
            # Check for internal duplicates only
            result = checker.check_internal_contamination(pairs)

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contamination check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Contamination check failed: {str(e)}")


@router.post("/datasets/{dataset_id}/split")
async def split_dataset(
    dataset_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    method: str = "hash",
):
    """
    Split a dataset into train/val/test splits.

    Args:
        dataset_id: Dataset to split
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        seed: Random seed for reproducibility
        method: "hash" (deterministic) or "random"

    Returns:
        Split information and updated dataset
    """
    from services.dataset_splitter import DatasetSplitter

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        splitter = DatasetSplitter()

        if method == "hash":
            result = splitter.split_by_hash(pairs, train_ratio, val_ratio, test_ratio, seed)
        else:
            result = splitter.split_by_ratio(pairs, train_ratio, val_ratio, test_ratio, seed)

        # Add split labels to pairs and update dataset
        labeled_pairs = splitter.assign_split_labels(
            pairs, train_ratio, val_ratio, test_ratio, seed, method
        )

        # Update dataset with labeled pairs
        dataset["pairs"] = labeled_pairs
        dataset["split_info"] = {
            "method": method,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_count": result.train.pair_count,
            "val_count": result.val.pair_count,
            "test_count": result.test.pair_count,
        }

        _save_dataset(dataset_id, dataset)

        return {
            "dataset_id": dataset_id,
            "splits": result.to_dict(),
            "message": f"Split {len(pairs)} pairs into train/val/test",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset split failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")


@router.post("/datasets/{dataset_id}/deduplicate")
async def deduplicate_dataset(
    dataset_id: str,
    method: str = "exact",
    similarity_threshold: float = 0.85,
):
    """
    Remove duplicate pairs from a dataset.

    Args:
        dataset_id: Dataset to deduplicate
        method: "exact" (hash-based) or "semantic" (embedding-based)
        similarity_threshold: For semantic method, similarity threshold

    Returns:
        Deduplication results and updated dataset
    """
    from services.dataset_validator import deduplicate_pairs, add_content_hashes

    try:
        dataset = _load_dataset(dataset_id)
        pairs = dataset.get("pairs", [])

        original_count = len(pairs)
        deduplicated, removed = deduplicate_pairs(pairs, method, similarity_threshold)

        # Add content hashes to deduplicated pairs
        deduplicated = add_content_hashes(deduplicated)

        # Update dataset
        dataset["pairs"] = deduplicated
        dataset["deduplication_info"] = {
            "method": method,
            "original_count": original_count,
            "deduplicated_count": len(deduplicated),
            "removed_count": len(removed),
        }

        _save_dataset(dataset_id, dataset)

        return {
            "dataset_id": dataset_id,
            "original_count": original_count,
            "deduplicated_count": len(deduplicated),
            "removed_count": len(removed),
            "removed_pairs": [
                {
                    "question": p.get("question", p.get("query", ""))[:100],
                    "duplicate_of": p.get("_duplicate_of"),
                }
                for p in removed[:20]  # Limit output
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")


# =============================================================================
# Background Evaluation Task Endpoints
# =============================================================================

@router.post("/tasks/start")
async def start_evaluation_task(request: RunEvaluationRequest):
    """
    Start an evaluation as a background task.

    Returns immediately with a task ID that can be used to track progress.
    The evaluation runs in the background and results are saved when complete.

    This is the recommended way to run evaluations as it:
    - Survives browser disconnections
    - Provides real-time progress updates
    - Can be cancelled if needed

    Supports two modes:
    1. Legacy mode: collection_name provided (use existing indexed collection)
    2. New mode: preprocessed_dataset_id + chunking config (on-demand embedding)
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()

    # Prepare chunking config dict if provided
    chunking_dict = None
    if request.chunking:
        chunking_dict = {
            "method": request.chunking.method,
            "chunk_size": request.chunking.chunk_size,
            "chunk_overlap": request.chunking.chunk_overlap,
        }

    task_id = await service.create_task(
        experiment_name=request.experiment_name,
        eval_dataset_id=request.eval_dataset_id,
        collection_name=request.collection_name,
        use_rag=request.use_rag,
        use_colbert=request.use_colbert,
        top_k=request.top_k,
        temperature=request.temperature,
        embedder=request.embedder,
        # New flow parameters
        preprocessed_dataset_id=request.preprocessed_dataset_id,
        preset=request.preset,
        chunking=chunking_dict,
    )

    return {"task_id": task_id, "message": "Evaluation started"}


@router.get("/tasks/{task_id}")
async def get_evaluation_task(task_id: str):
    """
    Get the status and progress of an evaluation task.

    Returns:
        Task details including:
        - status: pending, running, completed, failed, cancelled
        - current_pair: Current pair being evaluated
        - total_pairs: Total pairs to evaluate
        - progress_percent: Completion percentage
        - current_step: Human-readable current action
        - result_run_id: ID of saved run (when completed)
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    task = service.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task


@router.get("/tasks")
async def list_evaluation_tasks(limit: int = 20):
    """
    List recent evaluation tasks.

    Returns the most recent tasks ordered by creation time.
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    return service.list_tasks(limit=limit)


@router.post("/tasks/{task_id}/cancel")
async def cancel_evaluation_task(task_id: str):
    """
    Cancel a running evaluation task.

    Only running tasks can be cancelled. Completed or failed tasks
    cannot be cancelled.
    """
    from services.evaluation_task_service import get_evaluation_task_service

    service = get_evaluation_task_service()
    success = await service.cancel_task(task_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} cannot be cancelled (not running or not found)"
        )

    return {"message": f"Task {task_id} cancelled"}


# =============================================================================
# Experiment Presets & Embedding Cache Endpoints
# =============================================================================

@router.get("/presets")
async def list_experiment_presets():
    """
    List available experiment presets.

    Returns predefined configurations for quick setup:
    - Quick Test: Fast iteration with smaller chunks
    - Balanced: Good balance of quality and speed (recommended)
    - High Quality: Best accuracy with larger chunks and better embedder
    """
    return {
        name: {
            "name": preset.name,
            "description": preset.description,
            "chunking": {
                "method": preset.chunking.method,
                "chunk_size": preset.chunking.chunk_size,
                "chunk_overlap": preset.chunking.chunk_overlap,
            },
            "embedder": preset.embedder,
            "top_k": preset.top_k,
            "reranker": preset.reranker,
            "temperature": preset.temperature,
        }
        for name, preset in EXPERIMENT_PRESETS.items()
    }


@router.get("/embedding-cache")
async def list_embedding_caches(preprocessed_dataset_id: int = None):
    """
    List all cached embedding collections.

    Cached collections can be reused to avoid re-embedding when running
    experiments with the same configuration.
    """
    db_gen = get_db_session()
    db = next(db_gen)

    try:
        caches = embedding_service.list_cached_collections(
            db, preprocessed_dataset_id=preprocessed_dataset_id
        )
        return caches
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


@router.delete("/embedding-cache/{cache_id}")
async def delete_embedding_cache(cache_id: int):
    """
    Delete a cached embedding collection.

    This removes the cache entry. The vector store collection may still
    exist and should be cleaned up separately if needed.
    """
    db_gen = get_db_session()
    db = next(db_gen)

    try:
        deleted = embedding_service.delete_cached_collection(db, cache_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Cache {cache_id} not found")
        return {"status": "deleted", "id": cache_id}
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
