# vLLM Migration Refactoring Guide

## Executive Summary

This document outlines all mismatches, inconsistencies, and required changes after migrating from Ollama to vLLM. The migration is partially complete, but several critical issues remain that will cause runtime errors, along with misleading naming conventions and outdated documentation.

**Status**: 🔴 CRITICAL ISSUES PRESENT

**Key Findings**:
- 2 critical bugs that will cause runtime failures
- 4 misleading function/variable names referencing Ollama
- 3 configuration hardcoding issues
- 10+ documentation files with extensive Ollama references

---

## Priority 1: Critical Fixes (MUST FIX - Runtime Errors)

### 1.1 Health Router - Incorrect Import and Type Annotations

**File**: `backend/routers/health_router.py`

**Problem**: Imports `ChatOllama` from `langchain_ollama` package which:
- Is NOT in project dependencies (see `backend/pyproject.toml`)
- Will cause `ImportError` at runtime
- Wrong type annotation for vLLM-based chat model

**Current Code** (Lines 11, 28-34):
```python
from langchain_ollama import ChatOllama  # ❌ WRONG - package not installed

async def health_check(
    pg_connection: AsyncConnection = Depends(get_async_pg_connection),
    ollama_chat_for_rag: Optional[ChatOllama] = Depends(  # ❌ WRONG TYPE
        get_optional_ollama_chat_for_rag
    ),
) -> HealthCheckResponse:
    health_service = HealthService()
    return await health_service.perform_health_check(
        pg_connection, ollama_chat_for_rag
    )
```

**Required Changes**:
```python
from langchain_openai import ChatOpenAI  # ✅ CORRECT

async def health_check(
    pg_connection: AsyncConnection = Depends(get_async_pg_connection),
    vllm_chat_for_rag: Optional[ChatOpenAI] = Depends(  # ✅ CORRECT TYPE
        get_optional_vllm_chat_for_rag  # Can keep using alias for backward compat
    ),
) -> HealthCheckResponse:
    health_service = HealthService()
    return await health_service.perform_health_check(
        pg_connection, vllm_chat_for_rag
    )
```

**Impact**: High - Will crash on startup or first health check
**Files to modify**:
- `backend/routers/health_router.py` (line 11, 28, 34)
- `backend/services/health.py` (update parameter name and type hints)

---

### 1.2 NeMo Service - Missing OLLAMA_BASE_URL Configuration

**File**: `backend/services/nemo.py`

**Problem**: Imports `OLLAMA_BASE_URL` from config, but this variable doesn't exist in `config.py`

**Current Code** (Lines 15, 28):
```python
from config import OLLAMA_BASE_URL  # ❌ Variable doesn't exist!

class NeMoService:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL  # ❌ Will cause AttributeError
```

**Available in config.py** (Lines 35-36):
```python
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
```

**Required Changes**:
```python
from config import VLLM_BASE_URL  # ✅ CORRECT

class NeMoService:
    def __init__(self):
        self.base_url = VLLM_BASE_URL  # ✅ CORRECT
```

**Impact**: High - Will crash when NeMoService is instantiated
**Files to modify**: `backend/services/nemo.py` (lines 15, 28)

---

## Priority 2: Misleading Names (Functional but Confusing)

### 2.1 Model Router - Ollama Function Names

**File**: `backend/routers/model_router.py`

**Problem**: Function name and docstring reference Ollama but actually works with vLLM

**Current Code** (Lines 20-26):
```python
@router.get(
    "", response_model=list[OllamaModelInfo]  # Using backward compat alias
)
async def list_ollama_models_endpoint(  # ❌ MISLEADING NAME
    model_service: ModelService = Depends(get_model_service),
):
    """
    List available Ollama models.  # ❌ MISLEADING DOCSTRING
    """
```

**Recommended Changes**:
```python
@router.get(
    "", response_model=list[ModelInfo]  # Use primary name, not alias
)
async def list_vllm_models_endpoint(  # ✅ CLEAR NAME
    model_service: ModelService = Depends(get_model_service),
):
    """
    List available vLLM models via OpenAI-compatible API.

    Returns model information including name, size, and format from the vLLM server.
    """
```

**Impact**: Medium - Works but confuses developers
**Files to modify**: `backend/routers/model_router.py` (lines 10, 20, 22, 26)

---

### 2.2 Chat Service - Unused Ollama Method

**File**: `backend/services/chat.py`

**Problem**: Method named `_guardrails_ollama_stream` but not Ollama-specific

**Current Code** (Line 223):
```python
async def _guardrails_ollama_stream(  # ❌ MISLEADING - not Ollama specific
    self,
    model_name_for_guardrails: str,
    messages_payload: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
```

**Status**: Currently unused (no callers found)

**Options**:
1. **Remove entirely** if truly unused
2. **Rename** to `_guardrails_nemo_stream` if will be used
3. **Rename** to `_guardrails_stream` (generic)

**Impact**: Low - Not currently called
**Files to modify**: `backend/services/chat.py` (line 223)

---

## Priority 3: Backward Compatibility (Document & Review)

### 3.1 RAG Components Aliases

**File**: `backend/rag_components.py`

**Current Code** (Lines 200, 222):
```python
# Backward compatibility aliases
get_ollama_chat_for_rag = get_vllm_chat_for_rag  # Line 200
get_optional_ollama_chat_for_rag = get_optional_vllm_chat_for_rag  # Line 222
```

**Status**: ✅ INTENTIONAL - Good for backward compatibility

**Recommendation**: Add deprecation comments
```python
# Backward compatibility alias - DEPRECATED: Use get_vllm_chat_for_rag instead
# This will be removed in a future version
get_ollama_chat_for_rag = get_vllm_chat_for_rag

# Backward compatibility alias - DEPRECATED: Use get_optional_vllm_chat_for_rag instead
# This will be removed in a future version
get_optional_ollama_chat_for_rag = get_optional_vllm_chat_for_rag
```

**Impact**: Low - Add documentation
**Files to modify**: `backend/rag_components.py` (lines 200, 222)

---

### 3.2 Schema Aliases

**File**: `backend/schemas.py`

**Current Code** (Line 54):
```python
OllamaModelInfo = ModelInfo  # Backward compatibility alias
```

**Status**: ✅ INTENTIONAL - Good for backward compatibility

**Recommendation**: Add deprecation comment
```python
# Backward compatibility alias - DEPRECATED: Use ModelInfo instead
# This will be removed in a future version
OllamaModelInfo = ModelInfo
```

**Impact**: Low - Add documentation
**Files to modify**: `backend/schemas.py` (line 54)

---

## Priority 4: Configuration Issues

### 4.1 Guardrails Hardcoded URLs

**Files**:
- `backend/guardrails_config/math_assistant/config.yml`
- `backend/guardrails_config/bank_assistant/config.yml`
- `backend/guardrails_config/aviation_assistant/config.yml`

**Problem**: All configs hardcode `http://localhost:8000/v1` which won't work in Docker

**Current Code** (All three files):
```yaml
models:
  - type: main
    engine: openai
    model: meta-llama/Llama-3.2-3B-Instruct
    parameters:
      base_url: "http://localhost:8000/v1"  # ❌ HARDCODED - won't work in Docker
      api_key: "EMPTY"
```

**Issue**: In Docker Compose, backend service needs to use service name `http://vllm:8000/v1`

**Recommendation**:
```yaml
models:
  - type: main
    engine: openai
    model: meta-llama/Llama-3.2-3B-Instruct
    parameters:
      base_url: "${VLLM_BASE_URL}/v1"  # ✅ Use environment variable
      api_key: "EMPTY"
```

**Alternative**: Load base_url from environment in code when initializing guardrails

**Impact**: Medium - Affects Docker deployment
**Files to modify**: All 3 guardrails config.yml files

---

## Priority 5: Documentation Updates

### 5.1 Documentation Files with Ollama References

**Files requiring updates**:

| File | Sections with Ollama References | Severity |
|------|--------------------------------|----------|
| `README.md` | Lines 3, 23, 43-51 | High - First impression |
| `docs/README.md` | Multiple references | High |
| `docs/DEVELOPMENT.md` | Lines 26, 58-85, 129, 162-173 | High - Developer onboarding |
| `docs/TROUBLESHOOTING.md` | Lines 9, 34, 52-53, 159-247 | High - Active debugging |
| `docs/BACKEND-GUIDE.md` | Lines 40, 132, 288-289, 325-326 | Medium |
| `docs/GUARDRAILS-GUIDE.md` | Lines 35, 42-43, 97, 103-104 | Medium |
| `docs/RAG-SYSTEM.md` | Lines 115, 827 | Medium |
| `docs/ARCHITECTURE.md` | Unknown - needs review | Medium |
| `docs/DEPLOYMENT.md` | Unknown - needs review | High |
| `docs/API-REFERENCE.md` | Unknown - needs review | Medium |
| `deploy/README.md` | Lines 30, 33, 47-59, 71, 144, 155-156, 174-184 | High - Deployment |

**Key Changes Needed**:
- Replace "Ollama" with "vLLM" throughout
- Update installation instructions (remove Ollama install, add vLLM Docker info)
- Update API endpoint examples
- Update model listing examples
- Update troubleshooting steps

**Impact**: High - User and developer confusion
**Files to modify**: All documentation files listed above

---

### 5.2 Specific Documentation Examples

#### README.md Issues

**Line 3** - Wrong tech stack:
```markdown
❌ RAG system with Ollama integration
✅ RAG system with vLLM integration
```

**Lines 43-51** - Wrong prerequisites:
```markdown
❌ - **Ollama** (for LLM inference)
✅ - **Docker** (vLLM runs in container)
```

#### docs/DEVELOPMENT.md Issues

**Lines 58-85** - Wrong setup instructions:
```markdown
❌ ### Installing Ollama
❌ Download and install from [ollama.ai](https://ollama.ai)
✅ ### vLLM Setup
✅ vLLM runs in Docker container (see docker-compose.yml)
```

#### docs/TROUBLESHOOTING.md Issues

**Lines 159-247** - Entire "Ollama Issues" section needs replacement with "vLLM Issues"

---

## Summary Tables

### Critical Fixes Required

| File | Line(s) | Issue | Impact | Effort |
|------|---------|-------|--------|--------|
| `backend/routers/health_router.py` | 11, 28, 34 | Wrong import & type | Runtime crash | 5 min |
| `backend/services/health.py` | TBD | Parameter type update | Runtime crash | 5 min |
| `backend/services/nemo.py` | 15, 28 | Missing config var | Runtime crash | 2 min |

**Total Critical Fixes**: 3 files, ~15 minutes

---

### Code Cleanup Required

| File | Line(s) | Issue | Impact | Effort |
|------|---------|-------|--------|--------|
| `backend/routers/model_router.py` | 10, 20-26 | Misleading names | Confusion | 5 min |
| `backend/services/chat.py` | 223 | Misleading method name | Confusion | 2 min |
| `backend/rag_components.py` | 200, 222 | Missing deprecation docs | Minor | 2 min |
| `backend/schemas.py` | 54 | Missing deprecation docs | Minor | 1 min |

**Total Code Cleanup**: 4 files, ~10 minutes

---

### Configuration Updates Required

| File | Issue | Impact | Effort |
|------|-------|--------|--------|
| `backend/guardrails_config/*/config.yml` (3 files) | Hardcoded URLs | Docker broken | 10 min |

**Total Config Updates**: 3 files, ~10 minutes

---

### Documentation Updates Required

| Category | Files | Impact | Effort |
|----------|-------|--------|--------|
| High Priority | 6 files | User confusion | 2-3 hours |
| Medium Priority | 5 files | Developer confusion | 1-2 hours |

**Total Documentation**: 11+ files, ~4 hours

---

## Implementation Checklist

### Phase 1: Critical Fixes (DO FIRST) ⚠️

- [ ] **Fix health_router.py**
  - [ ] Line 11: Change import from `langchain_ollama` to `langchain_openai`
  - [ ] Line 28: Change type from `ChatOllama` to `ChatOpenAI`
  - [ ] Line 28: Rename parameter from `ollama_chat_for_rag` to `vllm_chat_for_rag`
  - [ ] Line 34: Update parameter name in function call

- [ ] **Fix services/health.py**
  - [ ] Update method signature to accept `ChatOpenAI` type
  - [ ] Update parameter name from `ollama_chat_for_rag` to `vllm_chat_for_rag`

- [ ] **Fix services/nemo.py**
  - [ ] Line 15: Change import from `OLLAMA_BASE_URL` to `VLLM_BASE_URL`
  - [ ] Line 28: Change usage from `OLLAMA_BASE_URL` to `VLLM_BASE_URL`

- [ ] **Test critical fixes**
  - [ ] Run health check endpoint
  - [ ] Verify NeMo service initialization
  - [ ] Check for import errors

### Phase 2: Code Cleanup

- [ ] **Update model_router.py**
  - [ ] Rename `list_ollama_models_endpoint` to `list_vllm_models_endpoint`
  - [ ] Update docstring to mention vLLM
  - [ ] Consider using `ModelInfo` directly instead of alias

- [ ] **Update chat.py**
  - [ ] Decide: Remove or rename `_guardrails_ollama_stream`
  - [ ] If renaming, use `_guardrails_stream` or `_guardrails_nemo_stream`

- [ ] **Add deprecation warnings**
  - [ ] Add comments to `rag_components.py` aliases (lines 200, 222)
  - [ ] Add comment to `schemas.py` alias (line 54)

### Phase 3: Configuration Updates

- [ ] **Fix guardrails configs**
  - [ ] `math_assistant/config.yml`: Replace hardcoded URL with env var
  - [ ] `bank_assistant/config.yml`: Replace hardcoded URL with env var
  - [ ] `aviation_assistant/config.yml`: Replace hardcoded URL with env var
  - [ ] Test guardrails initialization in Docker

### Phase 4: Documentation Updates

- [ ] **High Priority Docs**
  - [ ] `README.md`: Update tech stack, prerequisites, getting started
  - [ ] `docs/README.md`: Update overview and references
  - [ ] `docs/DEVELOPMENT.md`: Replace Ollama setup with vLLM info
  - [ ] `docs/TROUBLESHOOTING.md`: Replace Ollama section with vLLM section
  - [ ] `docs/DEPLOYMENT.md`: Update deployment instructions
  - [ ] `deploy/README.md`: Update deployment guide

- [ ] **Medium Priority Docs**
  - [ ] `docs/BACKEND-GUIDE.md`: Update LLM integration sections
  - [ ] `docs/GUARDRAILS-GUIDE.md`: Update model configuration examples
  - [ ] `docs/RAG-SYSTEM.md`: Update RAG pipeline references
  - [ ] `docs/ARCHITECTURE.md`: Update architecture diagrams/descriptions
  - [ ] `docs/API-REFERENCE.md`: Update API examples

### Phase 5: Verification

- [ ] **Code verification**
  - [ ] Search codebase for any remaining "ollama" references (case-insensitive)
  - [ ] Run type checker (`mypy` if configured)
  - [ ] Run linter
  - [ ] Run all tests

- [ ] **Runtime verification**
  - [ ] Start services with `docker-compose up`
  - [ ] Test health check endpoint
  - [ ] Test model listing endpoint
  - [ ] Test chat with guardrails enabled
  - [ ] Test RAG queries
  - [ ] Test document upload and processing

- [ ] **Documentation verification**
  - [ ] Review all updated docs for accuracy
  - [ ] Verify all commands/examples work as documented
  - [ ] Check for broken links

---

## Docker Compose Configuration (Reference)

**Current Status**: ✅ CORRECT

The `docker-compose.yml` is already properly configured:

```yaml
backend:
  environment:
    - VLLM_BASE_URL=http://vllm:8000  # ✅ Correct service reference

vllm:
  image: vllm/vllm-openai:latest
  environment:
    - MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct  # ✅ Correct model
  ports:
    - "8000:8000"
```

**No changes needed** in docker-compose.yml

---

## Dependencies Status (Reference)

**File**: `backend/pyproject.toml`

**Current Status**: ✅ CORRECT

```toml
dependencies = [
    "openai",              # ✅ For vLLM OpenAI-compatible API
    "langchain-openai",    # ✅ For LangChain integration with vLLM
    # NO langchain-ollama  # ✅ Correctly removed
]
```

**No changes needed** in pyproject.toml

---

## Testing Recommendations

After implementing fixes, test these scenarios:

1. **Health Check**: `curl http://localhost:8001/api/health`
2. **Model Listing**: `curl http://localhost:8001/api/models`
3. **Chat Request**: Send test message through API
4. **RAG Query**: Test document retrieval and response
5. **Guardrails**: Test with jailbreak attempt to verify NeMo integration

---

## Estimated Total Effort

| Phase | Time Estimate |
|-------|---------------|
| Phase 1: Critical Fixes | 30 minutes |
| Phase 2: Code Cleanup | 20 minutes |
| Phase 3: Configuration | 20 minutes |
| Phase 4: Documentation | 4 hours |
| Phase 5: Verification | 1 hour |
| **TOTAL** | **~6 hours** |

---

## Notes

- **Breaking Changes**: Renaming function parameters in health_router.py might affect API contracts if exposed
- **Backward Compatibility**: Current aliases allow old code to work, but should be deprecated with warnings
- **Documentation**: Most time-consuming part but critical for team alignment
- **Testing**: Ensure comprehensive testing after critical fixes before proceeding to cleanup phases

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: Ready for Implementation
