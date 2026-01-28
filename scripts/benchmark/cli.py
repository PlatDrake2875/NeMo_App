"""CLI interface for the benchmark tool."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from scripts.benchmark.client import BenchmarkAPIClient
import scripts.benchmark.commands.evaluate as evaluate_cmd
import scripts.benchmark.commands.generate_qa as generate_qa_cmd
import scripts.benchmark.commands.pipeline as pipeline_cmd
import scripts.benchmark.commands.preprocess as preprocess_cmd
from scripts.benchmark.config import (
    PRESETS,
    ChunkingConfig,
    CleaningConfig,
    EmbedderConfig,
    EvaluationConfig,
    FullBenchmarkConfig,
    LightweightMetadataConfig,
    LLMMetadataConfig,
    PreprocessingConfig,
    QAGenerationConfig,
)
from scripts.benchmark.utils.output import create_result_summary, export_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="benchmark",
    help="E2E Benchmarking CLI for RAG evaluation",
    add_completion=False,
)


def _load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _print_result(result: dict, as_json: bool = False) -> None:
    """Print result to stdout."""
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")


# =============================================================================
# PREPROCESS COMMAND
# =============================================================================


@app.command()
def preprocess(
    files: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--files",
            "-f",
            help="Files to upload (can specify multiple)",
            exists=True,
        ),
    ] = None,
    raw_dataset_id: Annotated[
        Optional[int],
        typer.Option(
            "--raw-dataset-id",
            "-r",
            help="Existing raw dataset ID to use",
        ),
    ] = None,
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Name for the dataset",
        ),
    ] = "Benchmark Dataset",
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="YAML config file for preprocessing",
            exists=True,
        ),
    ] = None,
    # Cleaning options
    cleaning: Annotated[
        bool,
        typer.Option(
            "--cleaning/--no-cleaning",
            help="Enable document cleaning",
        ),
    ] = False,
    remove_headers_footers: Annotated[
        bool,
        typer.Option(
            "--remove-headers-footers",
            help="Remove headers and footers",
        ),
    ] = False,
    # Metadata options
    extract_keywords: Annotated[
        bool,
        typer.Option(
            "--extract-keywords",
            help="Extract RAKE keywords",
        ),
    ] = False,
    extract_statistics: Annotated[
        bool,
        typer.Option(
            "--extract-statistics",
            help="Extract document statistics",
        ),
    ] = False,
    detect_language: Annotated[
        bool,
        typer.Option(
            "--detect-language",
            help="Detect document language",
        ),
    ] = False,
    # LLM metadata
    llm_metadata: Annotated[
        bool,
        typer.Option(
            "--llm-metadata",
            help="Enable LLM-based metadata extraction",
        ),
    ] = False,
    # Embedder
    embedder: Annotated[
        str,
        typer.Option(
            "--embedder",
            help="Embedder model name",
        ),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    # Backend
    vector_backend: Annotated[
        str,
        typer.Option(
            "--vector-backend",
            help="Vector store backend (pgvector|qdrant)",
        ),
    ] = "pgvector",
    # Server
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Timeout in seconds for processing",
        ),
    ] = 3600.0,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output result as JSON",
        ),
    ] = False,
) -> None:
    """
    Create a processed dataset from files or raw dataset.

    Stage 1 of the benchmark pipeline.
    """
    # Build config from options or load from file
    if config:
        cfg = _load_yaml_config(config)
        preprocessing_config = PreprocessingConfig(**cfg.get("preprocessing", {}))
        embedder_config = EmbedderConfig(**cfg.get("embedder_config", {}))
        if "source" in cfg:
            source = cfg["source"]
            if source.get("type") == "files":
                files = [Path(p) for p in source.get("file_paths", [])]
                name = source.get("dataset_name", name)
            elif source.get("type") == "existing_raw":
                raw_dataset_id = source.get("raw_dataset_id")
    else:
        # Build from CLI options
        cleaning_config = CleaningConfig(
            enabled=cleaning,
            remove_headers_footers=remove_headers_footers,
        )
        lightweight_config = LightweightMetadataConfig(
            enabled=extract_keywords or extract_statistics or detect_language,
            extract_rake_keywords=extract_keywords,
            extract_statistics=extract_statistics,
            detect_language=detect_language,
        )
        llm_config = LLMMetadataConfig(enabled=llm_metadata)
        preprocessing_config = PreprocessingConfig(
            cleaning=cleaning_config,
            lightweight_metadata=lightweight_config,
            llm_metadata=llm_config,
            chunking=None,  # Defer to eval time
        )
        embedder_config = EmbedderConfig(model_name=embedder)

    # Validate inputs
    if files is None and raw_dataset_id is None:
        typer.echo("Error: Must provide either --files or --raw-dataset-id", err=True)
        raise typer.Exit(1)

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            result = await preprocess_cmd.preprocess(
                client=client,
                name=name,
                files=files,
                raw_dataset_id=raw_dataset_id,
                preprocessing_config=preprocessing_config,
                embedder_config=embedder_config,
                vector_backend=vector_backend,
                timeout=timeout,
            )
            _print_result(create_result_summary(result), as_json=output_json)

    asyncio.run(run())


# =============================================================================
# GENERATE-QA COMMAND
# =============================================================================


@app.command("generate-qa")
def generate_qa(
    processed_dataset_id: Annotated[
        int,
        typer.Option(
            "--processed-dataset-id",
            "-p",
            help="Processed dataset ID to generate from",
        ),
    ],
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Name for the Q&A dataset",
        ),
    ] = "Generated Q&A Dataset",
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="YAML config file",
            exists=True,
        ),
    ] = None,
    use_vllm: Annotated[
        bool,
        typer.Option(
            "--use-vllm/--no-vllm",
            help="Use vLLM for generation",
        ),
    ] = True,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model to use for generation",
        ),
    ] = None,
    pairs_per_chunk: Annotated[
        int,
        typer.Option(
            "--pairs-per-chunk",
            help="Number of Q&A pairs per chunk (1-5)",
        ),
    ] = 2,
    max_chunks: Annotated[
        Optional[int],
        typer.Option(
            "--max-chunks",
            help="Maximum chunks to process",
        ),
    ] = 50,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="Generation temperature (0-1)",
        ),
    ] = 0.3,
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = None,
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output result as JSON",
        ),
    ] = False,
) -> None:
    """
    Generate Q&A evaluation dataset from a processed dataset.

    Stage 2 of the benchmark pipeline.
    """
    # Load from config if provided
    if config:
        cfg = _load_yaml_config(config)
        processed_dataset_id = cfg.get("processed_dataset_id", processed_dataset_id)
        name = cfg.get("name", name)
        use_vllm = cfg.get("use_vllm", use_vllm)
        model = cfg.get("model", model)
        pairs_per_chunk = cfg.get("pairs_per_chunk", pairs_per_chunk)
        max_chunks = cfg.get("max_chunks", max_chunks)
        temperature = cfg.get("temperature", temperature)
        seed = cfg.get("seed", seed)

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            result = await generate_qa_cmd.generate_qa(
                client=client,
                processed_dataset_id=processed_dataset_id,
                name=name,
                use_vllm=use_vllm,
                model=model,
                pairs_per_chunk=pairs_per_chunk,
                max_chunks=max_chunks,
                temperature=temperature,
                seed=seed,
            )
            _print_result(create_result_summary(result), as_json=output_json)

    asyncio.run(run())


# =============================================================================
# EVALUATE COMMAND
# =============================================================================


@app.command()
def evaluate(
    eval_dataset_id: Annotated[
        str,
        typer.Option(
            "--eval-dataset-id",
            "-e",
            help="Evaluation dataset ID",
        ),
    ],
    processed_dataset_id: Annotated[
        int,
        typer.Option(
            "--processed-dataset-id",
            "-p",
            help="Processed dataset ID",
        ),
    ],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="YAML config file",
            exists=True,
        ),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset",
            help="Evaluation preset (quick|balanced|high_quality)",
        ),
    ] = None,
    # Custom chunking (overrides preset)
    chunk_size: Annotated[
        Optional[int],
        typer.Option(
            "--chunk-size",
            help="Chunk size (100-10000)",
        ),
    ] = None,
    chunk_overlap: Annotated[
        Optional[int],
        typer.Option(
            "--chunk-overlap",
            help="Chunk overlap (0-2000)",
        ),
    ] = None,
    chunk_method: Annotated[
        str,
        typer.Option(
            "--chunk-method",
            help="Chunking method (recursive|fixed|semantic)",
        ),
    ] = "recursive",
    embedder: Annotated[
        str,
        typer.Option(
            "--embedder",
            help="Embedder model name",
        ),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    use_colbert: Annotated[
        bool,
        typer.Option(
            "--use-colbert/--no-colbert",
            help="Use ColBERT reranking",
        ),
    ] = False,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of chunks to retrieve (1-50)",
        ),
    ] = 5,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="Generation temperature (0-1)",
        ),
    ] = 0.1,
    experiment_name: Annotated[
        Optional[str],
        typer.Option(
            "--experiment-name",
            help="Name for the experiment",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output CSV file path",
        ),
    ] = None,
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Timeout in seconds",
        ),
    ] = 3600.0,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output result as JSON",
        ),
    ] = False,
) -> None:
    """
    Run evaluation on a Q&A dataset.

    Stage 3 of the benchmark pipeline.
    """
    # Load from config if provided
    if config:
        cfg = _load_yaml_config(config)
        eval_dataset_id = cfg.get("eval_dataset_id", eval_dataset_id)
        processed_dataset_id = cfg.get("processed_dataset_id", processed_dataset_id)
        preset = cfg.get("preset", preset)
        if "chunking" in cfg:
            chunking_cfg = cfg["chunking"]
            chunk_size = chunking_cfg.get("chunk_size", chunk_size)
            chunk_overlap = chunking_cfg.get("chunk_overlap", chunk_overlap)
            chunk_method = chunking_cfg.get("method", chunk_method)
        embedder = cfg.get("embedder", embedder)
        use_colbert = cfg.get("use_colbert", use_colbert)
        top_k = cfg.get("top_k", top_k)
        temperature = cfg.get("temperature", temperature)
        experiment_name = cfg.get("experiment_name", experiment_name)
        if "output_dir" in cfg and output is None:
            output = Path(cfg["output_dir"]) / "results.csv"

    # Build chunking config if custom options provided, otherwise resolve from preset
    chunking = None
    if chunk_size is not None or chunk_overlap is not None:
        chunking = ChunkingConfig(
            method=chunk_method,
            chunk_size=chunk_size or 1000,
            chunk_overlap=chunk_overlap or 200,
        )
    elif preset and preset in PRESETS:
        # Resolve preset to actual chunking config
        preset_config = PRESETS[preset]
        chunking = preset_config["chunking"]
        # Also apply other preset defaults if not explicitly overridden
        if embedder == "sentence-transformers/all-MiniLM-L6-v2":  # default value
            embedder = preset_config["embedder"]
        if top_k == 5:  # default value
            top_k = preset_config["top_k"]
        if not use_colbert:  # default is False
            use_colbert = preset_config["use_colbert"]

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            result = await evaluate_cmd.evaluate(
                client=client,
                eval_dataset_id=eval_dataset_id,
                processed_dataset_id=processed_dataset_id,
                preset=preset,
                chunking=chunking,
                embedder=embedder,
                use_colbert=use_colbert,
                top_k=top_k,
                temperature=temperature,
                experiment_name=experiment_name,
                output_path=output,
                timeout=timeout,
            )
            _print_result(create_result_summary(result), as_json=output_json)

    asyncio.run(run())


# =============================================================================
# PIPELINE COMMAND
# =============================================================================


@app.command()
def pipeline(
    config: Annotated[
        Path,
        typer.Argument(
            help="YAML config file for full pipeline",
            exists=True,
        ),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output result as JSON",
        ),
    ] = False,
) -> None:
    """
    Run full E2E benchmark pipeline.

    Runs all stages (preprocess, generate-qa, evaluate) from a config file.
    """
    cfg = _load_yaml_config(config)
    benchmark_config = FullBenchmarkConfig(**cfg)

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            result = await pipeline_cmd.pipeline(
                client=client,
                config=benchmark_config,
            )
            _print_result(create_result_summary(result), as_json=output_json)
            if not result.success:
                raise typer.Exit(1)

    asyncio.run(run())


# =============================================================================
# LIST COMMAND
# =============================================================================


@app.command("list")
def list_resources(
    type_: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Resource type (raw|processed|eval|runs)",
        ),
    ] = "processed",
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """List available datasets or runs."""

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            if type_ == "raw":
                resources = await client.list_raw_datasets()
                headers = ["ID", "Name", "Files", "Size"]
                rows = [
                    [r["id"], r["name"], r.get("total_file_count", 0), r.get("total_size_bytes", 0)]
                    for r in resources
                ]
            elif type_ == "processed":
                resources = await client.list_processed_datasets()
                headers = ["ID", "Name", "Status", "Chunks", "Raw ID"]
                rows = [
                    [
                        r["id"],
                        r["name"],
                        r.get("processing_status", "unknown"),
                        r.get("chunk_count", 0),
                        r.get("raw_dataset_id", ""),
                    ]
                    for r in resources
                ]
            elif type_ == "eval":
                resources = await client.list_eval_datasets()
                headers = ["ID", "Name", "Pairs", "Created"]
                rows = [
                    [
                        r["id"],
                        r["name"],
                        r.get("pair_count", len(r.get("pairs", []))),
                        r.get("created_at", "")[:19],
                    ]
                    for r in resources
                ]
            elif type_ == "runs":
                resources = await client.list_eval_runs()
                headers = ["ID", "Dataset", "Precision", "Recall", "Created"]
                rows = [
                    [
                        r["id"][:8] + "...",
                        r.get("eval_dataset_id", "")[:8] + "...",
                        f"{r.get('metrics', {}).get('precision_at_k', 0):.2%}",
                        f"{r.get('metrics', {}).get('recall_at_k', 0):.2%}",
                        r.get("created_at", "")[:19],
                    ]
                    for r in resources
                ]
            else:
                typer.echo(f"Unknown type: {type_}", err=True)
                raise typer.Exit(1)

            if output_json:
                print(json.dumps(resources, indent=2, default=str))
            else:
                # Print as table
                col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
                header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
                separator = "-+-".join("-" * w for w in col_widths)
                print(header_line)
                print(separator)
                for row in rows:
                    print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))

    asyncio.run(run())


# =============================================================================
# EXPORT COMMAND
# =============================================================================


@app.command()
def export(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            "-r",
            help="Evaluation run ID",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file path",
        ),
    ],
    format_: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (csv|json)",
        ),
    ] = "csv",
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="API server base URL",
        ),
    ] = "http://localhost:8000",
) -> None:
    """Export evaluation run results."""

    async def run() -> None:
        async with BenchmarkAPIClient(base_url=base_url) as client:
            if format_ == "csv":
                await client.export_run_csv(run_id, output)
                typer.echo(f"Exported CSV to {output}")
            elif format_ == "json":
                run_data = await client.get_eval_run(run_id)
                export_json(run_data, output)
                typer.echo(f"Exported JSON to {output}")
            else:
                typer.echo(f"Unknown format: {format_}", err=True)
                raise typer.Exit(1)

    asyncio.run(run())


# =============================================================================
# PRESETS COMMAND
# =============================================================================


@app.command()
def presets() -> None:
    """List available evaluation presets."""
    print("\nAvailable Presets:\n")
    for name, preset in PRESETS.items():
        chunking = preset["chunking"]
        print(f"  {name}:")
        print(f"    Chunking: {chunking.method}, size={chunking.chunk_size}, overlap={chunking.chunk_overlap}")
        print(f"    Embedder: {preset['embedder']}")
        print(f"    Top-K: {preset['top_k']}")
        print(f"    ColBERT: {preset['use_colbert']}")
        print()


if __name__ == "__main__":
    app()
