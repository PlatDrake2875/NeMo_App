#!/usr/bin/env python3
"""
CLI for RAG Evaluation operations.

Usage:
    python -m backend.cli.eval run --config eval_config.yaml
    python -m backend.cli.eval run --dataset my_dataset --collection my_collection
    python -m backend.cli.eval list-runs
    python -m backend.cli.eval list-runs --dataset my_dataset
    python -m backend.cli.eval export --run-id xyz --format csv
    python -m backend.cli.eval rescore --run-id xyz --metrics faithfulness,correctness
    python -m backend.cli.eval dry-run --config eval_config.yaml
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml

                return yaml.safe_load(f)
            except ImportError:
                print("Warning: PyYAML not installed, trying as JSON")
                return json.load(f)
        else:
            return json.load(f)


async def cmd_run(args):
    """Run an evaluation."""
    from services.eval_runner import EvalConfig, EvalRunner
    from services.eval_scorer import EvalScorer

    # Build config
    if args.config:
        config_data = load_config_file(args.config)
        config = EvalConfig.from_dict(config_data)
    else:
        config = EvalConfig(
            embedder_model=args.embedder or "sentence-transformers/all-MiniLM-L6-v2",
            reranker_strategy=args.reranker or "none",
            top_k=args.top_k or 5,
            temperature=args.temperature or 0.1,
            collection_name=args.collection or "rag_documents",
            use_rag=not args.no_rag,
        )

    # Progress callback
    def progress(current, total):
        print(f"\rProgress: {current}/{total} ({current / total * 100:.1f}%)", end="", flush=True)

    # Run evaluation
    runner = EvalRunner()
    print(f"Running evaluation with config hash: {config.config_hash()}")
    print(f"Dataset: {args.dataset or 'Quick Test'}")
    print(f"Collection: {config.collection_name}")
    print()

    run_result = await runner.run_evaluation(
        config=config,
        dataset_id=args.dataset,
        test_query=args.query,
        progress_callback=progress,
    )

    print()  # New line after progress

    # Score results
    scorer = EvalScorer()
    scored_result = await scorer.score_run(run_result)

    # Save results
    saved_path = scorer.save_scored_run(scored_result)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Run ID: {scored_result.run_id}")
    print(f"Config Hash: {config.config_hash()}")
    print(f"Pairs Evaluated: {scored_result.metrics.total_pairs}")
    print(f"Successful: {scored_result.metrics.successful_pairs}")
    print()
    print("METRICS:")
    print(f"  Context Precision:  {scored_result.metrics.context_precision * 100:.1f}%")
    print(f"  Precision@K:        {scored_result.metrics.precision_at_k * 100:.1f}%")
    print(f"  Recall@K:           {scored_result.metrics.recall_at_k * 100:.1f}%")
    print(f"  Avg Latency:        {scored_result.metrics.avg_latency:.2f}s")
    print()
    print(f"Results saved to: {saved_path}")


async def cmd_list_runs(args):
    """List evaluation runs."""
    runs_dir = Path("data/evaluation_runs")

    if not runs_dir.exists():
        print("No evaluation runs found.")
        return

    runs = []
    for path in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                runs.append({
                    "id": data.get("id", data.get("run_id", path.stem)),
                    "name": data.get("name", data.get("dataset_name", "Unknown")),
                    "created_at": data.get("created_at", ""),
                    "pairs": len(data.get("results", [])),
                    "config_hash": data.get("config", {}).get("config_hash", ""),
                    "precision_at_k": data.get("metrics", {}).get("precision_at_k", 0),
                    "recall_at_k": data.get("metrics", {}).get("recall_at_k", 0),
                })
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}", file=sys.stderr)

    if args.dataset:
        runs = [r for r in runs if args.dataset.lower() in r["name"].lower()]

    if not runs:
        print("No matching evaluation runs found.")
        return

    # Print as table
    print(f"{'ID':<10} {'Name':<30} {'Pairs':>6} {'P@K':>8} {'R@K':>8} {'Created':<20}")
    print("-" * 90)
    for run in runs[:args.limit]:
        print(
            f"{run['id']:<10} "
            f"{run['name'][:28]:<30} "
            f"{run['pairs']:>6} "
            f"{run['precision_at_k']*100:>7.1f}% "
            f"{run['recall_at_k']*100:>7.1f}% "
            f"{run['created_at'][:19]:<20}"
        )


async def cmd_export(args):
    """Export an evaluation run."""
    runs_dir = Path("data/evaluation_runs")
    run_path = runs_dir / f"{args.run_id}.json"

    if not run_path.exists():
        print(f"Run not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)

    with open(run_path) as f:
        data = json.load(f)

    output_path = args.output or f"evaluation_{args.run_id}.{args.format}"

    if args.format == "json":
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif args.format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "Query",
                "Predicted Answer",
                "Ground Truth",
                "Context Precision",
                "Precision@K",
                "Recall@K",
                "Latency (s)",
            ])

            # Data rows
            for result in data.get("results", []):
                scores = result.get("scores", {})
                writer.writerow([
                    result.get("query", ""),
                    result.get("predicted_answer", ""),
                    result.get("ground_truth", ""),
                    f"{(scores.get('context_precision', 0)) * 100:.1f}%",
                    f"{(scores.get('precision_at_k', 0)) * 100:.1f}%",
                    f"{(scores.get('recall_at_k', 0)) * 100:.1f}%",
                    f"{result.get('latency', 0):.2f}",
                ])

            # Summary
            metrics = data.get("metrics", {})
            writer.writerow([])
            writer.writerow(["SUMMARY"])
            writer.writerow(["Context Precision", f"{(metrics.get('context_precision') or 0) * 100:.1f}%"])
            writer.writerow(["Precision@K", f"{(metrics.get('precision_at_k') or 0) * 100:.1f}%"])
            writer.writerow(["Recall@K", f"{(metrics.get('recall_at_k') or 0) * 100:.1f}%"])
            writer.writerow(["Avg Latency", f"{(metrics.get('avg_latency') or 0):.2f}s"])

    print(f"Exported to: {output_path}")


async def cmd_rescore(args):
    """Re-score an evaluation run with new metrics."""
    from services.eval_scorer import EvalScorer

    scorer = EvalScorer()

    metrics = args.metrics.split(",") if args.metrics else None
    print(f"Re-scoring run {args.run_id} with metrics: {metrics or 'all'}")

    scored_result = await scorer.rescore_run(args.run_id, metrics)

    # Save if requested
    if args.save:
        saved_path = scorer.save_scored_run(scored_result)
        print(f"Saved re-scored results to: {saved_path}")

    # Print summary
    print()
    print("RESCORED METRICS:")
    print(f"  Context Precision:  {scored_result.metrics.context_precision * 100:.1f}%")
    print(f"  Precision@K:        {scored_result.metrics.precision_at_k * 100:.1f}%")
    print(f"  Recall@K:           {scored_result.metrics.recall_at_k * 100:.1f}%")


async def cmd_dry_run(args):
    """Validate configuration without running."""
    from services.eval_runner import EvalConfig, EvalRunner

    if args.config:
        config_data = load_config_file(args.config)
        config = EvalConfig.from_dict(config_data)
    else:
        config = EvalConfig(
            embedder_model=args.embedder or "sentence-transformers/all-MiniLM-L6-v2",
            reranker_strategy=args.reranker or "none",
            top_k=args.top_k or 5,
            temperature=args.temperature or 0.1,
            collection_name=args.collection or "rag_documents",
            use_rag=not args.no_rag,
        )

    runner = EvalRunner()
    result = runner.dry_run(config, args.dataset)

    print("DRY RUN RESULT")
    print("=" * 40)
    print(f"Valid: {result['valid']}")
    print(f"Config Hash: {result['config_hash']}")
    print(f"Dataset: {result.get('dataset_name', 'Quick Test')}")
    print(f"Pairs: {result.get('pair_count', 1)}")

    if result.get("existing_run_id"):
        print(f"Existing Run: {result['existing_run_id']}")

    if result["errors"]:
        print("\nERRORS:")
        for error in result["errors"]:
            print(f"  - {error}")

    if result["warnings"]:
        print("\nWARNINGS:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


async def cmd_list_metrics(args):
    """List available metrics."""
    from services.metrics import list_metrics
    from services.metrics.registry import get_metric_info

    metrics = list_metrics()

    print("AVAILABLE METRICS")
    print("=" * 60)
    for name in sorted(metrics):
        info = get_metric_info(name)
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Requires context: {info['requires_context']}")
        print(f"  Requires reference: {info['requires_reference']}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command
    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("--config", "-c", help="Path to config file (YAML/JSON)")
    run_parser.add_argument("--dataset", "-d", help="Evaluation dataset ID")
    run_parser.add_argument("--collection", help="Vector store collection name")
    run_parser.add_argument("--embedder", help="Embedding model name")
    run_parser.add_argument("--reranker", choices=["none", "colbert"], help="Reranker strategy")
    run_parser.add_argument("--top-k", type=int, help="Number of documents to retrieve")
    run_parser.add_argument("--temperature", type=float, help="LLM temperature")
    run_parser.add_argument("--query", "-q", help="Single test query (if no dataset)")
    run_parser.add_argument("--no-rag", action="store_true", help="Disable RAG")

    # list-runs command
    list_parser = subparsers.add_parser("list-runs", help="List evaluation runs")
    list_parser.add_argument("--dataset", "-d", help="Filter by dataset name")
    list_parser.add_argument("--limit", "-n", type=int, default=20, help="Max runs to show")

    # export command
    export_parser = subparsers.add_parser("export", help="Export an evaluation run")
    export_parser.add_argument("--run-id", "-r", required=True, help="Run ID to export")
    export_parser.add_argument("--format", "-f", choices=["json", "csv"], default="csv", help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file path")

    # rescore command
    rescore_parser = subparsers.add_parser("rescore", help="Re-score a run with new metrics")
    rescore_parser.add_argument("--run-id", "-r", required=True, help="Run ID to re-score")
    rescore_parser.add_argument("--metrics", "-m", help="Comma-separated list of metrics")
    rescore_parser.add_argument("--save", "-s", action="store_true", help="Save re-scored results")

    # dry-run command
    dry_parser = subparsers.add_parser("dry-run", help="Validate config without running")
    dry_parser.add_argument("--config", "-c", help="Path to config file (YAML/JSON)")
    dry_parser.add_argument("--dataset", "-d", help="Evaluation dataset ID")
    dry_parser.add_argument("--collection", help="Vector store collection name")
    dry_parser.add_argument("--embedder", help="Embedding model name")
    dry_parser.add_argument("--reranker", choices=["none", "colbert"], help="Reranker strategy")
    dry_parser.add_argument("--top-k", type=int, help="Number of documents to retrieve")
    dry_parser.add_argument("--temperature", type=float, help="LLM temperature")
    dry_parser.add_argument("--no-rag", action="store_true", help="Disable RAG")

    # list-metrics command
    subparsers.add_parser("list-metrics", help="List available metrics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "list-runs":
        asyncio.run(cmd_list_runs(args))
    elif args.command == "export":
        asyncio.run(cmd_export(args))
    elif args.command == "rescore":
        asyncio.run(cmd_rescore(args))
    elif args.command == "dry-run":
        asyncio.run(cmd_dry_run(args))
    elif args.command == "list-metrics":
        asyncio.run(cmd_list_metrics(args))


if __name__ == "__main__":
    main()
