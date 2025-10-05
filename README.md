# Brainnett

A production-ready RAG (Retrieval-Augmented Generation) system with decentralized, verifiable document storage using Ceramic Network and IPFS.

## Overview

This workspace integrates [Docling](https://github.com/DS4SD/docling) document processing with:
- **Ceramic Network** for signed, mutable document streams
- **IPFS** for decentralized content storage  
- **PostgreSQL + PGVector** for fast vector similarity search
- **OpenAI** for embeddings and LLM inference

### Key Features

 **Verifiable Provenance** - Every chunk signed with DID  
 **Mutable History** - Update documents while preserving audit trail  
 **Decentralized Storage** - Content on IPFS, metadata on Ceramic  
 **Fast Retrieval** - PGVector for sub-millisecond similarity search  
 **Multi-format Support** - PDF, DOCX, HTML, Audio via Docling  
 **Production Ready** - Docker, K8s, monitoring, security

## Workspace Structure

```
docling-dltknowlagebase/
├── docling-rag-agent/          # Python RAG application
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── ingest.py           # Main ingestion orchestrator
│   │   ├── chunker.py          # Document chunking
│   │   ├── embedder.py         # Embedding generation
│   │   └── ceramic_client.py   # Ceramic Network client
│   ├── sql/
│   │   ├── schema.sql          # PostgreSQL schema
│   │   └── ceramic_streams.sql # Ceramic mapping table
│   ├── utils/                  # Utilities (DB, models, etc.)
│   ├── rag_agent.py            # RAG query interface
│   └── test_ceramic_integration.py
│
├── ceramic-service/            # TypeScript Ceramic service
│   ├── src/
│   │   ├── index.ts            # Main entry point
│   │   ├── server.ts           # Express API server
│   │   ├── ceramic.ts          # Ceramic operations
│   │   ├── ipfs.ts             # IPFS pinning
│   │   └── config.ts           # Configuration & DID
│   ├── package.json
│   ├── Dockerfile
│   └── test.js
│
├── ceramic-sdk/                # Ceramic TypeScript SDK (monorepo)
│   ├── packages/               # SDK packages
│   └── apps/explorer/          # Ceramic explorer UI
│
├── docker-compose.ceramic.yml  # Docker Compose setup
├── CERAMIC_INTEGRATION.md      # Complete design doc (8k words)
├── QUICKSTART.md               # Setup guide (7k words)
└── DEPLOYMENT.md               # Production deployment (12k words)
```

## Quick Start

### Prerequisites

- **Node.js 18+** (for Ceramic service)
- **Python 3.9+** (for RAG agent)
- **PostgreSQL 14+** with PGVector
- **Docker & Docker Compose** (recommended)
- **Pinata Account** (or local IPFS node)

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone <repo-url>
cd docling-dltknowlagebase

# 2. Generate DID seed
echo "AGENT_DID_SEED_BASE64=$(openssl rand -base64 32)" >> .env

# 3. Add credentials to .env
cat << EOF >> .env
PINATA_JWT=your-pinata-jwt-token
OPENAI_API_KEY=your-openai-key
EOF

# 4. Start all services
docker-compose -f docker-compose.ceramic.yml up -d

# 5. Apply database schemas
docker exec docling-postgres psql -U raguser -d postgres -f /docker-entrypoint-initdb.d/schema.sql
docker exec docling-postgres psql -U raguser -d postgres -f /docker-entrypoint-initdb.d/ceramic_streams.sql

# 6. Verify services
curl http://localhost:3001/health  # Ceramic service
curl http://localhost:5101/ceramic/liveness  # Ceramic node

# 7. Ingest documents
cd docling-rag-agent
python -m ingestion.ingest --documents ./documents
```

### Option 2: Manual Setup

See [QUICKSTART.md](QUICKSTART.md) for detailed manual setup instructions.

## Documentation

| Document | Description |
|----------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | Get started in 10 minutes |
| **[CERAMIC_INTEGRATION.md](CERAMIC_INTEGRATION.md)** | Complete design & architecture |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Production deployment guide |
| **[ceramic-service/README.md](ceramic-service/README.md)** | Ceramic service API docs |
| **[docling-rag-agent/README.md](docling-rag-agent/README.md)** | RAG agent documentation |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Document Ingestion                    │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Docling Parser      │  Multi-format support
    │   PDF, DOCX, Audio    │  (PDF, Word, Audio, etc.)
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │   Semantic Chunker    │  Smart chunking with
    │   + Token Counter     │  overlap & boundaries
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐
    │  Embedding Generator  │  OpenAI embeddings
    │  (text-embedding-3)   │  1536 dimensions
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────────────────────────────┐
    │         PostgreSQL + PGVector                  │
    │  ┌──────────┐  ┌────────────────┐            │
    │  │ chunks   │  │ ceramic_streams │            │
    │  │ (vector) │  │  (mappings)     │            │
    │  └──────────┘  └────────────────┘            │
    └──────────┬──────────────────────┬─────────────┘
               │                      │
               │                      ▼
               │           ┌─────────────────────────┐
               │           │  Ceramic Service (TS)   │
               │           │  - Sign with DID        │
               │           │  - Create streams       │
               │           │  - Pin to IPFS          │
               │           └──────────┬──────────────┘
               │                      │
               │                      ▼
               │           ┌─────────────────────────┐
               │           │   Ceramic Network       │
               │           │   (TileDocument)        │
               │           │   - Signed streams      │
               │           │   - Commit history      │
               │           └──────────┬──────────────┘
               │                      │
               │                      ▼
               │           ┌─────────────────────────┐
               │           │    IPFS Network         │
               │           │    (Pinned Content)     │
               │           └─────────────────────────┘
               │
               ▼
    ┌───────────────────────┐
    │    RAG Query Engine   │
    │  - Vector search      │
    │  - Context assembly   │
    │  - LLM generation     │
    └───────────────────────┘
```

## Key Concepts

### Ceramic Streams

Each document chunk is stored as a **TileDocument** on Ceramic Network:

- **Signed** with agent DID (cryptographically verifiable)
- **Mutable** (can be updated while preserving history)
- **Queryable** via stream ID
- **Permanent** (immutable commit history)

### Storage Strategy

- **Small chunks** (<16KB): Stored directly in Ceramic
- **Large chunks** (≥16KB): Content on IPFS, CID in Ceramic
- **Metadata**: Always in Ceramic (source, timestamps, etc.)
- **Embeddings**: PostgreSQL for fast similarity search

### DID Authentication

Documents signed with Decentralized Identifiers (DIDs):

```typescript
// Each stream signed by agent DID
did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH
```

Benefits:
- Provenance tracking
- Multi-tenant support
- Key rotation capability
- Cryptographic verification

## Testing

### Test Ceramic Service

```bash
cd ceramic-service
pnpm install
pnpm test
```

### Test Python Integration

```bash
cd docling-rag-agent
python test_ceramic_integration.py
```

### End-to-End Test

```bash
# 1. Start services
docker-compose -f docker-compose.ceramic.yml up -d

# 2. Ingest test documents
python -m ingestion.ingest --documents ./test-docs

# 3. Query database
psql $DATABASE_URL -c "SELECT * FROM chunks_with_ceramic LIMIT 5;"

# 4. Verify streams
curl http://localhost:3001/api/ceramic/stream/<stream-id>
```

## 🔧 Configuration

### Environment Variables

**Ceramic Service** (`ceramic-service/.env`):
```bash
CERAMIC_URL=http://localhost:5101
AGENT_DID_SEED_BASE64=<32-byte-base64-seed>
IPFS_PINNING_SERVICE=pinata
PINATA_JWT=<your-pinata-jwt>
PORT=3001
```

**RAG Agent** (`docling-rag-agent/.env`):
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/db
OPENAI_API_KEY=<your-key>
CERAMIC_SERVICE_URL=http://localhost:3001
ENABLE_CERAMIC_STREAMS=true
```

See [.env.example](docling-rag-agent/.env.example) for complete configuration.

## Monitoring

### Health Checks

```bash
# Ceramic service
curl http://localhost:3001/health

# Ceramic node
curl http://localhost:5101/ceramic/liveness

# PostgreSQL
psql $DATABASE_URL -c "SELECT 1"
```

### Metrics (Future)

- Stream creation rate
- IPFS pin success/failure
- DID authentication status
- Query latency (p50, p95, p99)

See [DEPLOYMENT.md](DEPLOYMENT.md#monitoring-and-observability) for Prometheus/Grafana setup.

## Security

- ✅ **DID seed** stored in environment/KMS (never in code)
- ✅ **Network isolation** in Docker/Kubernetes
- ✅ **Signed streams** for authenticity
- ✅ **Audit trail** via Ceramic commits
- ✅ **Key rotation** support

See [DEPLOYMENT.md](DEPLOYMENT.md#security-setup) for production security setup.

## Deployment

### Docker Compose

```bash
docker-compose -f docker-compose.ceramic.yml up -d
```

### Kubernetes

```bash
kubectl apply -f k8s/ceramic-service.yaml
kubectl apply -f k8s/ceramic-one.yaml
```

### Systemd (Linux)

```bash
sudo cp ceramic-service.service /etc/systemd/system/
sudo systemctl enable ceramic-service
sudo systemctl start ceramic-service
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

## Use Cases

1. **Document Provenance**: Prove authenticity and ownership
2. **Compliance**: Immutable audit trail for regulations
3. **Multi-tenant RAG**: Separate DIDs per tenant
4. **Decentralized Knowledge**: Resistant to single points of failure
5. **Version Tracking**: See document evolution over time




- [Docling](https://github.com/DS4SD/docling) - Document processing
- [Ceramic Network](https://ceramic.network/) - Decentralized streams
- [IPFS](https://ipfs.tech/) - Content addressing
- [PGVector](https://github.com/pgvector/pgvector) - Vector search

