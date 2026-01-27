"""
Background Evaluation Task Service.

Provides async evaluation execution with progress tracking.
Tasks run in the background and persist state to database,
allowing the UI to safely disconnect and reconnect.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import POSTGRES_CONNECTION_STRING, VLLM_BASE_URL, VLLM_MODEL
from database_models import EvaluationTask
from enums import EvaluationTaskStatus
from schemas.evaluation import (
    AnswerCorrectnessDetail,
    ClaimVerificationDetail,
    EvalConfig,
    EvalMetrics,
    EvalResult,
    EvalResultScores,
    FaithfulnessDetail,
    RetrievedChunk,
)
from services.evaluation_metrics import RAGEvaluationMetrics
from rag_components import get_rag_context_prefix, format_docs
import httpx

logger = logging.getLogger(__name__)

# Directory for storing evaluation runs
EVAL_RUNS_DIR = Path("data/evaluation_runs")
EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DATASETS_DIR = Path("data/evaluation_datasets")


class EvaluationTaskService:
    """Service for managing background evaluation tasks."""

    _instance: Optional["EvaluationTaskService"] = None
    _running_tasks: Dict[str, asyncio.Task] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._engine = create_engine(POSTGRES_CONNECTION_STRING)
            self._session_maker = sessionmaker(bind=self._engine)
            self._running_tasks = {}
            self._initialized = True

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._session_maker()

    async def create_task(
        self,
        eval_dataset_id: Optional[str],
        collection_name: str,
        use_rag: bool = True,
        use_colbert: bool = True,
        top_k: int = 5,
        temperature: float = 0.1,
        embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
        experiment_name: Optional[str] = None,
    ) -> str:
        """
        Create a new evaluation task and start it in the background.

        Returns:
            Task ID for tracking progress
        """
        task_id = str(uuid.uuid4())

        # Load dataset to get pair count and name
        total_pairs = 1
        dataset_name = "Quick Test"

        if eval_dataset_id:
            dataset_path = EVAL_DATASETS_DIR / f"{eval_dataset_id}.json"
            if dataset_path.exists():
                with open(dataset_path) as f:
                    dataset_data = json.load(f)
                    total_pairs = len(dataset_data.get("pairs", []))
                    dataset_name = dataset_data.get("name", "Unknown")

        # Use custom experiment name if provided, otherwise fall back to dataset name
        display_name = experiment_name if experiment_name else dataset_name

        # Create task record in database
        config = {
            "experiment_name": experiment_name,
            "eval_dataset_id": eval_dataset_id,
            "collection_name": collection_name,
            "use_rag": use_rag,
            "use_colbert": use_colbert,
            "top_k": top_k,
            "temperature": temperature,
            "embedder": embedder,
        }

        with self.get_session() as session:
            task = EvaluationTask(
                id=task_id,
                config=config,
                status=EvaluationTaskStatus.PENDING.value,
                current_pair=0,
                total_pairs=total_pairs,
                current_step="Initializing...",
                eval_dataset_name=display_name,
                collection_display_name=collection_name,
            )
            session.add(task)
            session.commit()

        # Start background task
        asyncio_task = asyncio.create_task(self._run_evaluation(task_id))
        self._running_tasks[task_id] = asyncio_task

        logger.info(f"Created evaluation task {task_id} with {total_pairs} pairs")
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and progress."""
        with self.get_session() as session:
            task = session.query(EvaluationTask).filter_by(id=task_id).first()
            if task:
                return task.to_dict()
        return None

    def list_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent evaluation tasks."""
        with self.get_session() as session:
            tasks = (
                session.query(EvaluationTask)
                .order_by(EvaluationTask.created_at.desc())
                .limit(limit)
                .all()
            )
            return [t.to_dict() for t in tasks]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]

        with self.get_session() as session:
            task = session.query(EvaluationTask).filter_by(id=task_id).first()
            if task and task.status == EvaluationTaskStatus.RUNNING.value:
                task.status = EvaluationTaskStatus.CANCELLED.value
                task.completed_at = datetime.now(timezone.utc)
                task.current_step = "Cancelled by user"
                session.commit()
                logger.info(f"Cancelled evaluation task {task_id}")
                return True
        return False

    def _update_task_progress(
        self,
        task_id: str,
        current_pair: Optional[int] = None,
        current_step: Optional[str] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        result_run_id: Optional[str] = None,
    ):
        """Update task progress in database."""
        with self.get_session() as session:
            task = session.query(EvaluationTask).filter_by(id=task_id).first()
            if task:
                if current_pair is not None:
                    task.current_pair = current_pair
                if current_step is not None:
                    task.current_step = current_step
                if status is not None:
                    task.status = status
                    if status == EvaluationTaskStatus.RUNNING.value:
                        task.started_at = datetime.now(timezone.utc)
                    elif status in [
                        EvaluationTaskStatus.COMPLETED.value,
                        EvaluationTaskStatus.FAILED.value,
                    ]:
                        task.completed_at = datetime.now(timezone.utc)
                if error_message is not None:
                    task.error_message = error_message
                if result_run_id is not None:
                    task.result_run_id = result_run_id
                session.commit()

    async def _run_evaluation(self, task_id: str):
        """Run the evaluation in the background."""
        try:
            # Get task config
            task_dict = self.get_task(task_id)
            if not task_dict:
                logger.error(f"Task {task_id} not found")
                return

            config = task_dict["config"]
            experiment_name = config.get("experiment_name")
            eval_dataset_id = config.get("eval_dataset_id")
            collection_name = config["collection_name"]
            use_rag = config.get("use_rag", True)
            use_colbert = config.get("use_colbert", True)
            top_k = config.get("top_k", 5)
            temperature = config.get("temperature", 0.1)
            embedder = config.get("embedder", "sentence-transformers/all-MiniLM-L6-v2")

            # Update status to running
            self._update_task_progress(
                task_id,
                status=EvaluationTaskStatus.RUNNING.value,
                current_step="Loading dataset...",
            )

            # Load pairs
            if eval_dataset_id:
                dataset_path = EVAL_DATASETS_DIR / f"{eval_dataset_id}.json"
                with open(dataset_path) as f:
                    dataset_data = json.load(f)
                    pairs = dataset_data.get("pairs", [])
                    dataset_name = dataset_data.get("name", "Unknown")
            else:
                pairs = [
                    {
                        "query": "What information is available in the documents?",
                        "ground_truth": "Based on retrieved documents.",
                    }
                ]
                dataset_name = "Quick Test"

            # Use custom experiment name if provided
            run_name = experiment_name if experiment_name else dataset_name

            total_pairs = len(pairs)
            self._update_task_progress(task_id, current_step=f"Processing {total_pairs} pairs...")

            # Initialize metrics evaluator
            metrics_evaluator = RAGEvaluationMetrics()

            results = []
            total_relevancy = 0.0
            total_faithfulness = 0.0
            total_correctness = 0.0
            total_context_precision = 0.0
            total_latency = 0.0

            for i, pair in enumerate(pairs):
                # Check if task was cancelled
                task_dict = self.get_task(task_id)
                if task_dict and task_dict["status"] == EvaluationTaskStatus.CANCELLED.value:
                    logger.info(f"Task {task_id} was cancelled, stopping evaluation")
                    return

                query = pair.get("query", pair.get("question", ""))
                ground_truth = pair.get("ground_truth", pair.get("expected_answer", ""))

                self._update_task_progress(
                    task_id,
                    current_pair=i + 1,
                    current_step=f"Evaluating: {query[:50]}...",
                )

                start_time = time.time()

                try:
                    # Get RAG context
                    if use_rag:
                        rag_prefix = await get_rag_context_prefix(
                            query,
                            collection_name=collection_name,
                            use_colbert=use_colbert,
                            embedder=embedder,
                        )
                        prompt_content = rag_prefix if rag_prefix else query
                    else:
                        prompt_content = query

                    # Generate answer
                    predicted_answer, retrieved_chunks = await self._generate_answer(
                        prompt_content, query, temperature, use_rag
                    )

                    generation_latency = time.time() - start_time
                    chunk_contents = [c.content for c in retrieved_chunks] if retrieved_chunks else []

                    # Calculate metrics
                    jaccard_score = self._calculate_jaccard(predicted_answer, ground_truth)
                    correctness_result = await metrics_evaluator.answer_correctness(
                        predicted_answer, ground_truth
                    )
                    faithfulness_result = await metrics_evaluator.faithfulness(
                        predicted_answer, chunk_contents
                    )
                    relevancy = metrics_evaluator.response_relevancy(query, predicted_answer)
                    precision_result = await metrics_evaluator.context_precision(
                        query, chunk_contents, ground_truth
                    )

                    total_latency += generation_latency
                    total_relevancy += relevancy
                    total_faithfulness += faithfulness_result.score
                    total_correctness += correctness_result.score
                    total_context_precision += precision_result.score

                    # Build result
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
                            score=jaccard_score,
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
                    logger.error(f"Error evaluating pair {i + 1}: {e}", exc_info=True)
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
            n = len([r for r in results if r.scores is not None])
            metrics = EvalMetrics(
                answer_relevancy=total_relevancy / n if n > 0 else 0.0,
                faithfulness=total_faithfulness / n if n > 0 else 0.0,
                context_precision=total_context_precision / n if n > 0 else 0.0,
                answer_correctness=total_correctness / n if n > 0 else 0.0,
                avg_latency=total_latency / len(results) if results else 0.0,
            )

            # Save evaluation run
            run_id = str(uuid.uuid4())[:8]
            eval_config = EvalConfig(
                collection=collection_name,
                use_rag=use_rag,
                use_colbert=use_colbert,
                top_k=top_k,
                temperature=temperature,
                embedder=embedder,
                llm_model=VLLM_MODEL,
            )

            run_data = {
                "id": run_id,
                "name": run_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_dataset_id": eval_dataset_id,
                "config": eval_config.model_dump(),
                "metrics": metrics.model_dump(),
                "results": [r.model_dump() for r in results],
            }

            run_path = EVAL_RUNS_DIR / f"{run_id}.json"
            with open(run_path, "w") as f:
                json.dump(run_data, f, indent=2, default=str)

            # Update task as completed
            self._update_task_progress(
                task_id,
                current_pair=total_pairs,
                status=EvaluationTaskStatus.COMPLETED.value,
                current_step="Evaluation complete",
                result_run_id=run_id,
            )

            logger.info(f"Evaluation task {task_id} completed, run saved as {run_id}")

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            self._update_task_progress(
                task_id,
                status=EvaluationTaskStatus.CANCELLED.value,
                current_step="Cancelled",
            )
        except Exception as e:
            logger.error(f"Evaluation task {task_id} failed: {e}", exc_info=True)
            self._update_task_progress(
                task_id,
                status=EvaluationTaskStatus.FAILED.value,
                current_step="Failed",
                error_message=str(e),
            )
        finally:
            # Clean up running task reference
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]

    async def _generate_answer(
        self,
        prompt_content: str,
        original_query: str,
        temperature: float,
        use_rag: bool,
    ) -> tuple[str, Optional[List[RetrievedChunk]]]:
        """Generate an answer using vLLM."""
        messages = [{"role": "user", "content": prompt_content}]

        payload = {
            "model": VLLM_MODEL,
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

            # Extract retrieved chunks from prompt
            retrieved_chunks = None
            if use_rag and "Context:" in prompt_content:
                try:
                    context_start = prompt_content.find("Context:") + len("Context:")
                    question_start = prompt_content.find("Question:")
                    if question_start > context_start:
                        context_text = prompt_content[context_start:question_start].strip()
                        chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
                        retrieved_chunks = [
                            RetrievedChunk(content=chunk[:500], source="retrieved")
                            for chunk in chunks[:5]
                        ]
                except Exception:
                    pass

            return answer, retrieved_chunks

    def _calculate_jaccard(self, predicted: str, ground_truth: str) -> float:
        """Calculate Jaccard similarity."""
        pred_words = set(predicted.lower().split())
        truth_words = set(ground_truth.lower().split())

        if not pred_words or not truth_words:
            return 0.0

        intersection = pred_words & truth_words
        union = pred_words | truth_words

        return len(intersection) / len(union) if union else 0.0


# Singleton instance
_service: Optional[EvaluationTaskService] = None


def get_evaluation_task_service() -> EvaluationTaskService:
    """Get the evaluation task service singleton."""
    global _service
    if _service is None:
        _service = EvaluationTaskService()
    return _service
