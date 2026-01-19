# RAG Benchmark Hub - Code Review Report

## Executive Summary

This document presents a comprehensive code review of the RAG Benchmark Hub component within the NeMo App. The review focuses on evaluation reproducibility, metric extensibility, dataset management, and frontend user experience.

---

## Current Architecture Analysis

### Backend Components

#### 1. Evaluation Router (`backend/routers/evaluation_router.py`)
**Strengths:**
- Clean API design with well-documented endpoints
- Proper async/await patterns for LLM calls
- JSON file persistence for evaluation runs

**Improvement Opportunities:**
- Evaluation configuration is embedded in run logic (not reusable)
- No separation between inference and scoring phases
- Missing batch evaluation support
- No dry-run capability for configuration validation

#### 2. Evaluation Metrics (`backend/services/evaluation_metrics.py`)
**Strengths:**
- RAGAS-style metrics implementation
- LLM-based claim extraction and verification
- Embedding similarity using sentence-transformers

**Improvement Opportunities:**
- All metrics hardcoded in single file
- No plugin architecture for custom metrics
- Cannot add new metrics without code changes

#### 3. Schemas (`backend/schemas/evaluation.py`)
**Strengths:**
- Pydantic models for type safety
- Good coverage of evaluation concepts

**Improvement Opportunities:**
- QAPair lacks fields for difficulty, answer types, source tracking
- No config hashing for reproducibility

### Frontend Components

#### 1. EvaluationPage (`frontend/src/.../evaluation/EvaluationPage.jsx`)
**Strengths:**
- Functional configuration UI
- Basic results table with expandable rows
- Evaluation history with load/export

**Improvement Opportunities:**
- No run comparison functionality
- No metric visualizations (charts)
- No annotation/editing capabilities
- Missing dataset version tracking

---

## Recommendations

### Phase 1: Evaluation Reproducibility

1. **EvalConfig as First-Class Entity**
   - Add deterministic config hashing
   - Store config with each run
   - Enable config presets

2. **Separate Inference from Scoring**
   - `EvalRunner`: Handles RAG pipeline execution
   - `EvalScorer`: Computes metrics from stored results
   - Benefits: Re-score runs with new metrics

### Phase 2: Metric Plugin System

1. **Registry Pattern**
   - BaseMetric abstract class
   - `@register_metric` decorator
   - Dynamic metric discovery

2. **Refactor Existing Metrics**
   - Extract to individual files
   - Maintain backward compatibility

### Phase 3: Dataset Improvements

1. **Enhanced QA Schema**
   - `alternative_answers`: Multiple valid answers
   - `answer_type`: extractive, abstractive, yes_no, multi_hop
   - `difficulty`: easy, medium, hard
   - `source_chunk_ids`: Traceability

2. **Dataset Versioning** (Deferred)
   - Preserve old versions on reprocessing
   - API support for version selection

3. **Standard Format Import**
   - SQuAD, Natural Questions, MS MARCO
   - Validation on import

### Phase 4: API & CLI

1. **Batch Evaluation**
   - Submit multiple configs
   - Parallel execution

2. **CLI Interface**
   - `python -m backend.cli.eval run`
   - `python -m backend.cli.eval export`

3. **Dry-Run Mode**
   - Validate config without execution
   - Report if new run would be created

### Phase 5: Frontend Redesign

1. **New Components**
   - EvaluationDashboard with overview stats
   - EvalConfigBuilder wizard
   - EvalRunComparison view
   - ResultsCharts with visualizations
   - AnnotationWorkbench for human review

2. **Comparison Features**
   - Side-by-side metric comparison
   - Per-question diff view
   - Statistical significance highlighting

3. **Visualizations**
   - Radar charts for metric comparison
   - Distribution histograms
   - Scatter plots for pattern analysis

---

## Implementation Priority

### Parallel Workstreams

| Workstream | Focus Area | Key Deliverables |
|------------|------------|------------------|
| A | Backend Evaluation | EvalConfig, Runner/Scorer separation, Batch API |
| B | Metrics & Datasets | Plugin registry, Importers, Validator |
| C | Frontend | Dashboard, Comparison, Annotation UI |
| D | Infrastructure | Typed config, CLI |

### Deferred (Backlog)

- Database migration for eval results (keep JSON for now)
- Dataset versioning in database
- Alembic migrations for new tables

---

## Files Summary

### New Files (~40)
- Backend: services/eval_runner.py, eval_scorer.py, metrics/*, dataset_importers/*, cli/*
- Frontend: evaluation/*, annotation/*, datasets/*, shared/*

### Modified Files (~10)
- Backend: schemas/evaluation.py, routers/evaluation_router.py, config.py
- Frontend: EvaluationPage.jsx, App.jsx

---

## Success Criteria

1. **Reproducibility**: Same config always produces same results
2. **Extensibility**: Add new metrics without code changes
3. **Usability**: Compare runs, visualize results, annotate data
4. **Performance**: Batch evaluation, parallel execution

---

*Report generated from comprehensive code review*
*Date: January 2026*
