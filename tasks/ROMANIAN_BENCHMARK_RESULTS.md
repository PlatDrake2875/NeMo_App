# Romanian Embedder Benchmark - Experiment Results

**Generated:** 2026-01-30

## Configuration

| Parameter | Value |
|-----------|-------|
| Embedder | `BlackKakapo/stsb-xlm-r-multilingual-ro` |
| LLM Model | `meta-llama/Llama-3.2-1B-Instruct` |
| RAG Enabled | Yes |
| ColBERT Reranking | No |
| Top-K | 5 |
| Temperature | 0.1 |
| Chunking | Recursive, 1500 chars, 300 overlap |

---

## Aggregate Metrics

| Dataset | Pairs | Answer Correctness | Faithfulness | Context Precision | Answer Relevancy | Avg Latency |
|---------|-------|-------------------|--------------|-------------------|------------------|-------------|
| Carte de Bucate | 52 | 27.5% | 29.9% | **9.9%** | 73.4% | 5.26s |
| Constitutia Romaniei | 59 | **46.9%** | **53.9%** | **30.8%** | **78.2%** | 6.82s |
| Istoria Romanilor | 63 | 27.2% | 43.8% | 26.7% | 51.4% | 7.37s |
| **Weighted Average** | **174** | **34.0%** | **43.1%** | **23.1%** | **67.0%** | **6.55s** |

---

## Failure Analysis

| Dataset | Zero Context Precision | Zero Faithfulness | Placeholder Answers |
|---------|----------------------|-------------------|---------------------|
| Carte de Bucate | 77% (40/52) | 40% (21/52) | 0% |
| Constitutia Romaniei | 58% (34/59) | 24% (14/59) | 7% (4/59) |
| Istoria Romanilor | 62% (39/63) | 24% (15/63) | 10% (6/63) |

---

## Run Identifiers

| Dataset | Run ID | Eval Dataset ID |
|---------|--------|-----------------|
| Carte de Bucate | `8a3378ed` | `5c0d40ab` |
| Constitutia Romaniei | `a188cc0f` | `1e9185e6` |
| Istoria Romanilor | `2285d91e` | `22be566a` |

---

## Key Findings

### 1. Retrieval is the Bottleneck
- **60-77% of queries have zero context precision**
- The Romanian embedder struggles to retrieve relevant chunks
- This directly impacts faithfulness and correctness scores

### 2. Best Performer: Constitutia Romaniei
- Highest correctness (46.9%) and faithfulness (53.9%)
- Legal/constitutional text is more structured
- Formal language matches better with embeddings

### 3. Worst Performer: Carte de Bucate (Cookbook)
- Lowest context precision (9.9%)
- PDF contains complex formatting, tables, ingredient lists
- OCR artifacts in extracted text (e.g., "Ńet", "ămâie")

### 4. Language Inconsistency
- LLM sometimes answers in English when ground truth is Romanian
- Example: Expected `"1/4 de litru de apă"` vs Got `"1/4 liter of water"`
- This causes semantic similarity mismatch

### 5. Q&A Generation Issues
- 7-10% of pairs have placeholder answers: `"The answer based on the text"`
- These are impossible to match correctly
- Need stricter validation in Q&A generation

---

## Sample Failures

### Carte de Bucate - Lowest Scoring
```
Q: What is the typical preparation method for the salată de fasole boabe?
Expected: Fierbinte, amestec cu sosul de o Ńet și ulei...
Got: Based on the context, the typical preparation method...
Scores: Correctness 3.9%, Faithfulness 0%, Context Precision 0%
```

### Constitutia - Lowest Scoring
```
Q: What is the main difference between pluralism and political parties?
Expected: The answer based on the text [PLACEHOLDER]
Got: Based on the context provided, the main difference...
Scores: Correctness 2.5%, Faithfulness 0%, Context Precision 0%
```

---

## Recommendations

### Immediate Improvements
1. **Enable ColBERT reranking** - should improve context precision
2. **Add Romanian language instruction** to LLM system prompt
3. **Regenerate Q&A pairs** with validation to reject placeholders

### Future Experiments
1. Try `intfloat/multilingual-e5-large` embedder
2. Test with larger LLM (`Llama-3.2-3B-Instruct` or `8B`)
3. Improve PDF preprocessing pipeline
4. Compare with English documents as baseline

---

## Export Commands

```bash
# Export to CSV
curl http://localhost:8000/api/evaluation/runs/8a3378ed/csv > carte-bucate-results.csv
curl http://localhost:8000/api/evaluation/runs/a188cc0f/csv > constitutia-results.csv
curl http://localhost:8000/api/evaluation/runs/2285d91e/csv > istorie-results.csv
```
