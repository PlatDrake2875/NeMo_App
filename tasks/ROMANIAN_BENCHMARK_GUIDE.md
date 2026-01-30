# Romanian Embedder Benchmark Experiments

## Overview
This guide explains how to run RAG benchmark experiments on Romanian documents using the `BlackKakapo/stsb-xlm-r-multilingual-ro` embedder, which is specifically trained for Romanian semantic similarity.

**Current Status**: Infrastructure is running, PDFs are downloaded. Start from **Step 1**.

---

## Available Test Documents

Located in `data/eval_pdfs/`:
| File | Description |
|------|-------------|
| `carte-bucate-sanda-marin.pdf` | Romanian Cookbook - Sanda Marin (255 pages) |
| `constitutia-romaniei.pdf` | Romanian Constitution (2003 revision) |
| `istoria-romanilor.pdf` | History of Romanians |

---

## Step 1: Import PDFs as Raw Datasets

Before processing, PDFs must be imported as raw datasets.

### Option A: Via API (curl)

```bash
# Import each PDF as a raw dataset
curl -X POST "http://localhost:8000/api/evaluation/import/pdf-as-raw?dataset_id=carte-bucate-sanda-marin"
curl -X POST "http://localhost:8000/api/evaluation/import/pdf-as-raw?dataset_id=constitutia-romaniei"
curl -X POST "http://localhost:8000/api/evaluation/import/pdf-as-raw?dataset_id=istoria-romanilor"
```

### Option B: Via Web UI (Recommended)

1. Go to **RAG Benchmark Hub** > **Evaluation** > **Import Eval Datasets**
2. Select a PDF from the "Romanian PDFs" section
3. Click "Import as Raw Dataset"
4. Repeat for each PDF

---

## Step 2: Create Processed Datasets

For each raw dataset, create processed datasets with different configurations.

### Via API (curl)

```bash
# First, list raw datasets to get IDs
curl http://localhost:8000/api/raw-datasets | python3 -m json.tool

# Create processed dataset (replace RAW_DATASET_ID)
# Example: Baseline config with Romanian embedder
curl -X POST "http://localhost:8000/api/processed-datasets?start_processing=true" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "carte-bucate-baseline",
    "description": "Carte de Bucate - Baseline preprocessing",
    "raw_dataset_id": 1,
    "embedder_config": {
      "model_name": "BlackKakapo/stsb-xlm-r-multilingual-ro",
      "model_type": "huggingface"
    },
    "preprocessing_config": {
      "chunking": {
        "method": "recursive",
        "chunk_size": 1500,
        "chunk_overlap": 300
      },
      "cleaning": {
        "enabled": true
      }
    },
    "vector_backend": "qdrant"
  }'
```

### Via Web UI (Recommended)

1. Go to **RAG Benchmark Hub** > **Data Management**
2. Click on a raw dataset
3. Click **Create Processed Dataset**
4. Configure:
   - **Name**: e.g., `carte-bucate-baseline`
   - **Embedder**: `BlackKakapo/stsb-xlm-r-multilingual-ro`
   - **Chunking**: Recursive, 1500 chars, 300 overlap
5. Click **Create & Process**

---

## Step 3: Generate Q&A Evaluation Datasets

For each processed dataset, generate Q&A pairs using the LLM.

### Via API

```bash
# List processed datasets to get IDs
curl http://localhost:8000/api/processed-datasets | python3 -m json.tool

# Generate Q&A pairs (replace PROCESSED_ID)
curl -X POST http://localhost:8000/api/evaluation/generate-qa \
  -H "Content-Type: application/json" \
  -d '{
    "processed_dataset_id": 1,
    "dataset_name": "qa-carte-bucate-baseline",
    "max_pairs": 50,
    "pairs_per_chunk": 2
  }'

# Check generation status
curl "http://localhost:8000/api/evaluation/generate-qa/status/JOB_ID"
```

### Via Web UI (Recommended)

1. Go to **RAG Benchmark Hub** > **Evaluation** > **Generate Q&A**
2. Select a processed dataset
3. Configure:
   - **Max pairs**: 50-100
   - **Pairs per chunk**: 2
4. Click **Generate**

---

## Step 4: Run Evaluations

Run evaluations with different configurations to compare results.

### Available Presets

| Preset | Chunk Size | Overlap | Embedder | Top-K | ColBERT |
|--------|-----------|---------|----------|-------|---------|
| `quick` | 500 | 50 | all-MiniLM-L6-v2 | 3 | No |
| `balanced` | 1000 | 200 | all-MiniLM-L6-v2 | 5 | No |
| `high_quality` | 1500 | 300 | BAAI/bge-small-en-v1.5 | 7 | Yes |

### Via API

```bash
# List eval datasets to get IDs
curl http://localhost:8000/api/evaluation/datasets | python3 -m json.tool

# Run evaluation with a preset
curl -X POST http://localhost:8000/api/evaluation/tasks/start \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "romanian-baseline-test",
    "eval_dataset_id": "EVAL_DATASET_ID",
    "preprocessed_dataset_id": 1,
    "preset": "balanced",
    "embedder": "BlackKakapo/stsb-xlm-r-multilingual-ro",
    "use_rag": true,
    "use_colbert": false,
    "top_k": 5,
    "temperature": 0.1
  }'

# Check task progress
curl http://localhost:8000/api/evaluation/tasks/TASK_ID
```

### Via Web UI (Recommended)

1. Go to **RAG Benchmark Hub** > **Evaluation** > **Run Evaluation**
2. Select the evaluation dataset
3. Select the preprocessed dataset
4. Choose a preset or configure manually
5. Override embedder to `BlackKakapo/stsb-xlm-r-multilingual-ro`
6. Click **Start Evaluation**

---

## Step 5: View and Export Results

### Via API

```bash
# List all evaluation runs
curl http://localhost:8000/api/evaluation/runs | python3 -m json.tool

# Get specific run details
curl http://localhost:8000/api/evaluation/runs/RUN_ID | python3 -m json.tool

# Export to CSV
curl http://localhost:8000/api/evaluation/runs/RUN_ID/csv > results.csv
```

### Via CLI

```bash
# List runs
python -m backend.cli.eval list-runs

# Export run to CSV
python -m backend.cli.eval export --run-id RUN_ID --format csv --output results.csv
```

### Via Web UI

1. Go to **RAG Benchmark Hub** > **Evaluation** > **Results**
2. Click on a run to view details
3. Use **Export CSV** button to download

---

## Experiment Matrix

For a complete benchmark, create these combinations:

| PDF | Config | Embedder | Description |
|-----|--------|----------|-------------|
| carte-bucate | baseline | Romanian | Minimal preprocessing |
| carte-bucate | balanced | Romanian | Standard preprocessing |
| carte-bucate | high_quality | Romanian | Full preprocessing + reranking |
| constitutia | baseline | Romanian | ... |
| constitutia | balanced | Romanian | ... |
| ... | ... | ... | ... |

---

## Metrics Explanation

The evaluation produces these RAGAS-style metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Answer Correctness** | F1 factual + semantic similarity to ground truth | 0-1 |
| **Faithfulness** | Claims in answer supported by retrieved context | 0-1 |
| **Context Precision** | Quality of retrieved chunks ranking | 0-1 |
| **Answer Relevancy** | Embedding similarity between query and answer | 0-1 |

---

## Troubleshooting

### GPU Out of Memory
- Reduce `--gpu-memory-utilization` in vLLM config
- Use CPU embeddings by setting device in backend config

### vLLM 400 Errors
- Context too long: reduce chunk size
- Check vLLM logs: `docker logs nemo_app-vllm-gpu-1`

### Embedder Loading Fails
- Ensure HuggingFace model is accessible
- Check GPU memory availability

### Processing Stuck
- Check backend logs: `docker logs nemo_app-backend-1`
- Verify database connection: `docker logs nemo_app-postgres-1`

---

## Appendix: Initial Setup (Already Completed)

<details>
<summary>Click to expand setup steps (not needed if infrastructure is running)</summary>

### Prerequisites
- Docker & Docker Compose installed
- NVIDIA GPU with CUDA support (for vLLM)
- ~20GB disk space for models and data

### Clone Repository
```bash
git clone https://github.com/PlatDrake2875/NeMo_App.git
cd NeMo_App
```

### Download Romanian PDF Datasets
```bash
mkdir -p data/eval_pdfs
cd data/eval_pdfs

# Romanian Cookbook - Sanda Marin (255 pages)
curl -L -o carte-bucate-sanda-marin.pdf \
  "https://apiardeal.ro/biblioteca/carti/GASTRONOMIE/Carte_de_bucate_-_Sanda_Marin_-_255_pag.pdf"

# Romanian Constitution (2003 revision)
curl -L -o constitutia-romaniei.pdf \
  "https://www.ccr.ro/wp-content/uploads/2020/03/Constitutia-2003.pdf"

# History of Romanians
curl -L -o istoria-romanilor.pdf \
  "https://www.sociouman-usamvb.ro/documents/Elemente_de_istorie_a_Romaniei.pdf"

cd ../..
```

### Start Infrastructure
```bash
# Start postgres and backend
docker compose up -d postgres backend

# Wait for backend health
sleep 30
curl http://localhost:8000/health/live

# Start vLLM (GPU required)
docker compose --profile gpu up -d vllm-gpu

# Wait for vLLM to load model (~2 minutes)
sleep 120
curl http://localhost:8002/v1/models
```

</details>
