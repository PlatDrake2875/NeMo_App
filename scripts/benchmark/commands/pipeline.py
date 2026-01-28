"""Pipeline command - run full E2E benchmark."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from scripts.benchmark.client import BenchmarkAPIClient
from scripts.benchmark.commands.evaluate import evaluate
from scripts.benchmark.commands.generate_qa import generate_qa
from scripts.benchmark.commands.preprocess import preprocess
from scripts.benchmark.config import (
    ExistingProcessedSourceConfig,
    ExistingRawSourceConfig,
    FilesSourceConfig,
    FullBenchmarkConfig,
    PipelineResult,
)
from scripts.benchmark.utils.output import export_json, generate_output_path

logger = logging.getLogger(__name__)


async def pipeline(
    client: BenchmarkAPIClient,
    config: FullBenchmarkConfig,
    progress_callback: Callable[[dict], None] | None = None,
) -> PipelineResult:
    """
    Run a full E2E benchmark pipeline.

    This chains all three stages together:
    1. Preprocess: Create processed dataset from source
    2. Generate Q&A: Create evaluation dataset from processed dataset
    3. Evaluate: Run evaluation and collect metrics

    Args:
        client: API client instance
        config: Full benchmark configuration
        progress_callback: Optional callback for progress updates

    Returns:
        PipelineResult with all stage results and metrics
    """
    start_time = time.time()
    logger.info(f"Starting benchmark pipeline: {config.name}")

    try:
        # =====================================================================
        # Stage 1: Preprocessing
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 1: PREPROCESSING")
        logger.info("=" * 60)

        source = config.source

        if isinstance(source, FilesSourceConfig):
            # Create from files
            preprocess_result = await preprocess(
                client=client,
                name=source.dataset_name,
                files=source.file_paths,
                preprocessing_config=config.preprocessing,
                embedder_config=config.embedder_config,
                vector_backend=config.vector_backend,
                progress_callback=progress_callback,
            )
        elif isinstance(source, ExistingRawSourceConfig):
            # Create from existing raw dataset
            preprocess_result = await preprocess(
                client=client,
                name=config.name,
                raw_dataset_id=source.raw_dataset_id,
                preprocessing_config=config.preprocessing,
                embedder_config=config.embedder_config,
                vector_backend=config.vector_backend,
                progress_callback=progress_callback,
            )
        elif isinstance(source, ExistingProcessedSourceConfig):
            # Skip preprocessing, use existing processed dataset
            logger.info(f"Using existing processed dataset: {source.processed_dataset_id}")
            dataset_info = await client.get_processed_dataset(source.processed_dataset_id)
            preprocess_result = type(
                "PreprocessResult",
                (),
                {
                    "raw_dataset_id": dataset_info.get("raw_dataset_id", 0),
                    "processed_dataset_id": source.processed_dataset_id,
                    "document_count": dataset_info.get("document_count", 0),
                    "chunk_count": dataset_info.get("chunk_count", 0),
                },
            )()
        else:
            raise ValueError(f"Unknown source type: {type(source)}")

        logger.info(
            f"Preprocessing complete: processed_dataset_id={preprocess_result.processed_dataset_id}"
        )

        # =====================================================================
        # Stage 2: Q&A Generation
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 2: Q&A GENERATION")
        logger.info("=" * 60)

        qa_config = config.qa_generation
        qa_result = await generate_qa(
            client=client,
            processed_dataset_id=preprocess_result.processed_dataset_id,
            name=qa_config.name,
            use_vllm=qa_config.use_vllm,
            model=qa_config.model,
            pairs_per_chunk=qa_config.pairs_per_chunk,
            max_chunks=qa_config.max_chunks,
            temperature=qa_config.temperature,
            seed=qa_config.seed,
            system_prompt=qa_config.system_prompt,
        )

        logger.info(f"Q&A generation complete: eval_dataset_id={qa_result.eval_dataset_id}")

        # =====================================================================
        # Stage 3: Evaluation
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STAGE 3: EVALUATION")
        logger.info("=" * 60)

        eval_config = config.evaluation
        output_path = generate_output_path(config.output_dir, f"{config.name}_results", "csv")

        eval_result = await evaluate(
            client=client,
            eval_dataset_id=qa_result.eval_dataset_id,
            processed_dataset_id=preprocess_result.processed_dataset_id,
            preset=eval_config.preset,
            chunking=eval_config.chunking,
            embedder=eval_config.embedder,
            use_rag=eval_config.use_rag,
            use_colbert=eval_config.use_colbert,
            top_k=eval_config.top_k,
            temperature=eval_config.temperature,
            experiment_name=eval_config.experiment_name or config.name,
            output_path=output_path,
            progress_callback=progress_callback,
        )

        logger.info(f"Evaluation complete: run_id={eval_result.run_id}")

        # =====================================================================
        # Complete
        # =====================================================================
        duration = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE in {duration:.1f}s")
        logger.info("=" * 60)

        # Import here to avoid circular import
        from scripts.benchmark.config import (
            EvaluateResult as EvalResultModel,
            GenerateQAResult as QAResultModel,
            PreprocessResult as PreprocessResultModel,
        )

        # Convert to proper models if needed
        if not isinstance(preprocess_result, PreprocessResultModel):
            preprocess_result = PreprocessResultModel(
                raw_dataset_id=preprocess_result.raw_dataset_id,
                processed_dataset_id=preprocess_result.processed_dataset_id,
                document_count=preprocess_result.document_count,
                chunk_count=preprocess_result.chunk_count,
            )

        result = PipelineResult(
            name=config.name,
            preprocess=preprocess_result,
            qa_generation=qa_result,
            evaluation=eval_result,
            duration_seconds=duration,
            success=True,
        )

        # Export full results JSON
        json_path = generate_output_path(config.output_dir, f"{config.name}_full", "json")
        export_json(result, json_path)

        return result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Pipeline failed after {duration:.1f}s: {e}")

        # Return partial result with error
        from scripts.benchmark.config import (
            EvaluateResult as EvalResultModel,
            EvalMetrics,
            GenerateQAResult as QAResultModel,
            PreprocessResult as PreprocessResultModel,
        )

        return PipelineResult(
            name=config.name,
            preprocess=PreprocessResultModel(
                raw_dataset_id=0,
                processed_dataset_id=0,
            ),
            qa_generation=QAResultModel(
                eval_dataset_id="",
                name="",
                pair_count=0,
                chunks_processed=0,
            ),
            evaluation=EvalResultModel(
                run_id="",
                metrics=EvalMetrics(),
            ),
            duration_seconds=duration,
            success=False,
            error=str(e),
        )
