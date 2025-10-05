# Brainnett V2 - Proposed Features & Enhancements

**Branch**: holesky  
**Status**: Proposal / Feature Branch  
**Date**: October 5, 2025

---

## 🎯 Overview

This branch represents a clean slate for **Brainnett V2** — a major enhancement to the production-ready RAG system. The V2 release focuses on **embedding provider flexibility**, **cost optimization**, and **local/free model support** while maintaining full backwards compatibility with the existing OpenAI-based workflow.

**Key Innovation**: Replace or supplement expensive OpenAI embedding APIs with free, locally-hosted models using **Sentence Transformers** and **sparse encoders**, reducing operational costs by 5-10x for high-volume document ingestion.

---

## 🆕 Major Features in V2

### 1. **Multi-Provider Embedding Architecture**

**Problem Solved**: Current system is locked into OpenAI's embedding API, resulting in high costs for large-scale document ingestion and lack of offline capability.

**Solution**: Introduce a pluggable `EmbeddingProvider` abstraction that supports:
- ✅ **OpenAI** (existing, default for backwards compatibility)
- ✅ **Sentence Transformers** (local dense embeddings: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- ✅ **Sparse Encoders** (SPLADE models: `naver/splade-v3` for interpretable retrieval)
- ✅ **Pydantic AI** (future: agentic embedding workflows)

**Benefits**:
- **Cost Savings**: $20-130/1M embeddings (OpenAI) → ~$5-15 (local GPU compute)
- **Privacy**: Keep sensitive documents on-premise, no API calls
- **Speed**: 200-1000+ embeddings/sec on GPU vs. API rate limits
- **Offline**: No internet dependency for embedding generation
- **Customization**: Fine-tune models on domain-specific data

---

### 2. **GPU Acceleration & Hardware Optimization**

**Hardware Support**:
- **NVIDIA CUDA**: Primary GPU backend (GeForce, Tesla, A100)
- **Apple Silicon (MPS)**: M1/M2/M3 GPU acceleration
- **CPU Fallback**: Graceful degradation for environments without GPU

**Performance Targets**:
- Sentence Transformers on GPU: **200-1000+ embeddings/sec** (batch size 32-128)
- Sparse encoders on GPU: **50-200 embeddings/sec** (batch size 8-16)
- CPU-only mode: **50+ embeddings/sec** (acceptable for low-volume)

**Memory Efficiency**:
- Models: 500MB-2GB GPU/RAM per model
- Batch processing to maximize GPU utilization
- Model quantization support (int8, fp16) for 2-4x memory savings

---

### 3. **Architectural Design Patterns**

The V2 implementation follows industry-standard software engineering patterns:

#### **Partitioning (Task Decomposition)**
- Clear separation of ingestion pipeline stages: parse → chunk → embed → index → serve
- Embedding stage decomposed into: provider selection, model loading, batching, normalization, caching
- Each stage independently testable and replaceable

#### **Communication (Inter-Module Contracts)**
- Abstract `EmbeddingProvider` interface defines clear contract:
  ```python
  class EmbeddingProvider(ABC):
      @abstractmethod
      def embed_texts(texts: List[str], batch_size: int) -> List[List[float]]
      @abstractmethod
      def health_check() -> bool
      @property
      @abstractmethod
      def dimensions() -> int
  ```
- Provider-agnostic: callers don't know whether embeddings come from OpenAI, local models, or remote services

#### **Agglomeration (Combining Providers)**
- `EnsembleEmbeddingProvider`: Combine multiple providers for quality/robustness
- `FallbackProvider`: Primary provider fails → automatic failover to secondary
- A/B testing: Route % of traffic to new provider, compare metrics

#### **Mapping (Format Normalization)**
- Unified vector dimensionality (OpenAI: 1536, Sentence Transformers: 384/768, SPLADE: 30,522)
- Normalization layer handles padding, truncation, L2 normalization
- Database schema tracks provider metadata per embedding

#### **Decomposition (Modularity)**
- Small, testable modules: `providers/`, `utils/embedding_cache.py`, `utils/device_manager.py`
- Function-level decomposition for batching, caching, error handling
- Clear separation of concerns

---

### 4. **Intelligent Caching & Performance**

**Embedding Cache**:
- LRU cache with configurable size (default: 10,000 embeddings)
- Keyed by text hash + provider config
- **60-80% cache hit rate** on typical workloads
- Reduces redundant API calls and compute by 5x

**Batching Optimizer**:
- Dynamic batch sizing based on:
  - Provider limits (OpenAI: 8,191 inputs, local models: GPU memory)
  - Text length distribution
  - GPU memory availability
- Progress tracking for large ingestion jobs

**Cost Optimization**:
| Provider | 1M Embeddings | Latency (batch=32) | Quality (NDCG@10) |
|----------|---------------|-------------------|-------------------|
| OpenAI (small) | $20-50 | 300ms | 1.0 (baseline) |
| OpenAI (large) | $130 | 400ms | 1.0+ (best) |
| Sentence Transformers (MiniLM) | ~$5 (GPU) | 50ms | 0.95-0.98 |
| Sentence Transformers (mpnet) | ~$10 (GPU) | 80ms | 0.97-0.99 |
| SPLADE (sparse) | ~$15 (GPU) | 150ms | 0.90-0.95 |

---

### 5. **Fallback & Robustness**

**Health Check Monitoring**:
- Periodic health checks (every 60s) for each provider
- Automatic detection of model loading failures, API outages, GPU OOM

**Automatic Failover**:
```python
# Example: Primary local model → Fallback to OpenAI
primary = SentenceTransformerProvider(model="all-MiniLM-L6-v2")
fallback = OpenAIProvider(model="text-embedding-3-small")
provider = FallbackProvider(primary, fallback)
```

**Circuit Breaker**:
- After N consecutive failures, stop calling provider for T seconds
- Prevents cascading failures and resource exhaustion

---

### 6. **Database Schema Evolution**

**New Columns** (backwards compatible):
```sql
ALTER TABLE chunks ADD COLUMN embedding_provider VARCHAR(64) DEFAULT 'openai';
ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(128);
ALTER TABLE chunks ADD COLUMN embedding_dimensions INT DEFAULT 1536;
ALTER TABLE chunks ADD COLUMN embedding_created_at TIMESTAMP DEFAULT NOW();
```

**Benefits**:
- Track which provider generated each embedding
- Support mixed providers in same database
- Enable A/B testing and gradual migration
- Audit trail for compliance

---

### 7. **Migration Strategy**

**Phase 1: Alpha (Week 1-2)**
- Implement provider abstraction + Sentence Transformers
- Unit tests with small test models
- Internal staging deployment

**Phase 2: Beta (Week 3-4)**
- Opt-in `EMBEDDING_PROVIDER=sentence-transformers` flag
- Migration script for re-embedding existing documents
- Early adopter feedback

**Phase 3: Production Rollout (Week 5-6)**
- Gradual rollout: 10% → 50% → 100% of traffic
- Monitor metrics: latency, error rate, retrieval quality
- Rollback plan if quality degrades >5%

**Backwards Compatibility**:
- Default provider remains `openai` (no breaking changes)
- Existing embeddings work unchanged
- Migration is optional

---

## 📊 Expected Impact

### Cost Savings (Example: 10M Documents)

**Current (OpenAI only)**:
- Embeddings: 10M * $0.03 = **$300,000/year**
- API rate limits may require tiered plan

**V2 (Local Models)**:
- Embeddings: 10M * $0.005 (GPU compute) = **$50,000/year**
- **Savings: $250,000/year (83% reduction)**
- No API rate limits (controlled by GPU capacity)

### Performance Improvements

- **Throughput**: 2,000 emb/sec (OpenAI) → 500-1000 emb/sec (local GPU)
- **Latency**: 300ms (API) → 50ms (local, no network overhead)
- **Cache hit rate**: 60-80% → effectively 2,500-5,000 emb/sec
- **Offline capability**: Zero → 100% (full offline support)

### Quality Validation

- Sentence Transformers: **≥95% of OpenAI quality** (NDCG@10 on evaluation datasets)
- Sparse encoders: **≥90% quality** (specialized domains)
- Ensemble providers: **potential >100% quality** (multi-model consensus)

---

## 🛠️ Implementation Details

### Key Files & Modules

```
docling-rag-agent/
├── ingestion/
│   ├── embedder.py (updated)          # Factory pattern, provider registry
│   └── providers/                      # NEW
│       ├── base.py                     # EmbeddingProvider abstract class
│       ├── openai_provider.py          # Refactored from embedder.py
│       ├── sentence_transformer_provider.py  # NEW
│       ├── sparse_encoder_provider.py  # NEW
│       └── pydantic_ai_provider.py     # FUTURE
├── utils/
│   ├── embedding_cache.py              # NEW - LRU cache
│   ├── embedding_normalizer.py         # NEW - dimension mapping
│   ├── device_manager.py               # NEW - GPU/CPU selection
│   └── batching.py                     # NEW - dynamic batching
├── sql/
│   └── migrations/
│       └── 002_add_embedding_metadata.sql  # NEW
└── tests/
    ├── test_embedding_providers.py     # NEW
    └── benchmark_providers.py          # NEW
```

### Configuration (`.env`)

```bash
# Embedding Provider Configuration
EMBEDDING_PROVIDER=sentence-transformers  # openai | sentence-transformers | sparse-encoder | pydantic-ai
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=auto  # auto | cuda | mps | cpu
EMBEDDING_BATCH_SIZE=32
EMBEDDING_NORMALIZE=true
EMBEDDING_CACHE_SIZE=10000

# Fallback Configuration
EMBEDDING_FALLBACK_ENABLED=true
EMBEDDING_FALLBACK_PROVIDER=openai

# OpenAI (if using openai provider)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 🔬 Technical Deep Dive

### Sentence Transformers Integration

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads on first use, ~100MB-500MB)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Encode batch of texts
texts = ["Document chunk 1...", "Document chunk 2...", ...]
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,  # L2 normalization for cosine similarity
    convert_to_numpy=True
)
# Shape: (len(texts), 384) for all-MiniLM-L6-v2
```

### Sparse Encoder Example

```python
from sentence_transformers import SparseEncoder

# Load sparse model
model = SparseEncoder("naver/splade-v3", device="cuda")

# Encode documents (different from queries)
docs = ["This is a document...", ...]
doc_embeddings = model.encode_document(docs, batch_size=16)
# Shape: (len(docs), 30522) - sparse representation, >99% zeros

# Check sparsity
stats = SparseEncoder.sparsity(doc_embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")  # e.g., 99.84%
```

---

## 📖 Documentation

For complete implementation details, see:
- **`docling-dltknowlagebase/Brainnett_V2_CHANGELOG.md`** - Full technical specification
- **`docling-dltknowlagebase/QUICKSTART.md`** - Setup instructions
- **`docling-dltknowlagebase/DEPLOYMENT.md`** - Production deployment

---

## 🎯 Success Criteria

**Functional**:
- ✅ Users can set `EMBEDDING_PROVIDER=sentence-transformers` and run end-to-end ingestion
- ✅ Mixed providers coexist in same database (metadata tracked per embedding)
- ✅ Automatic fallback works (local model failure → OpenAI)
- ✅ Cache hit rate >60% on typical workloads

**Performance**:
- ✅ Sentence Transformers on GPU: >200 embeddings/sec
- ✅ Retrieval quality ≥95% of OpenAI baseline (NDCG@10)
- ✅ End-to-end latency <2x vs. OpenAI-only

**Operational**:
- ✅ Documentation updated with GPU setup instructions
- ✅ Migration guide for existing OpenAI users
- ✅ Tests: unit (>80% coverage), integration, load tests

---

## 🚀 Next Steps

1. **Review**: Team review of architectural design and implementation plan
2. **Prototype**: Build MVP with Sentence Transformers provider
3. **Benchmark**: Run performance and quality tests vs. OpenAI baseline
4. **Alpha**: Deploy to staging, gather feedback
5. **Beta**: Opt-in rollout to early adopters
6. **GA**: General availability with full documentation

---

## 📞 Contact & Collaboration

This proposal branch (`holesky`) serves as a clean starting point for V2 development discussions. No legacy git history is included to keep the focus on the future roadmap.

For questions or contributions to this proposal:
- Open issues in the main repository
- Review the full changelog in `docling-dltknowlagebase/Brainnett_V2_CHANGELOG.md`
- Join community discussions

---

**Built with a vision for cost-effective, privacy-preserving, and high-performance RAG systems**
