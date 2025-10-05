# Ceramic Network Integration - Quick Start Guide

This guide will help you set up and run the Ceramic Network integration with Docling RAG.

## Prerequisites

1. **Node.js 18+** (for Ceramic service)
2. **Python 3.9+** (for Docling RAG agent)
3. **PostgreSQL with PGVector** (for vector storage)
4. **Ceramic One node** (local or remote)
5. **IPFS Pinning Service** (Pinata account or local IPFS)

## Setup Steps

### 1. Database Setup

First, apply the Ceramic streams schema to your PostgreSQL database:

```bash
cd docling-rag-agent
psql $DATABASE_URL -f sql/ceramic_streams.sql
```

This creates the `ceramic_streams` table and view for tracking Ceramic stream mappings.

### 2. Generate DID Seed

Generate a secure 32-byte seed for your agent's DID:

```bash
openssl rand -base64 32
```

Copy the output - you'll need it in the next step.

### 3. Configure Environment Variables

Update your `.env` file in `docling-rag-agent/`:

```bash
# Add these to your existing .env

# Ceramic Network Configuration
CERAMIC_URL=http://localhost:5101
CERAMIC_SERVICE_URL=http://localhost:3001

# DID Configuration (paste your generated seed here)
AGENT_DID_SEED_BASE64=<your-generated-seed>

# IPFS Pinning Service
IPFS_PINNING_SERVICE=pinata

# Pinata Credentials (get from https://pinata.cloud)
PINATA_JWT=your-pinata-jwt-token

# Ceramic Integration Options
ENABLE_CERAMIC_STREAMS=true
CERAMIC_BATCH_SIZE=10
CERAMIC_RETRY_ATTEMPTS=3
```

Also create `.env` in `ceramic-service/`:

```bash
cd ../ceramic-service
cp .env.example .env
# Edit .env with your credentials
```

### 4. Install Dependencies

#### Ceramic SDK and Service (TypeScript)

```bash
cd ceramic-sdk
pnpm install

cd ../ceramic-service
pnpm install
```

#### Python RAG Agent

```bash
cd ../docling-rag-agent
pip install -e .
# or with uv:
uv pip install -e .
```

### 5. Start Ceramic One Node

You need a running Ceramic One node. Options:

**Option A: Local Ceramic One (Recommended for Development)**

```bash
# From ceramic-sdk directory
pnpm run c1
```

This starts a local Ceramic One node on port 5101.

**Option B: Use Remote Ceramic Node**

Set `CERAMIC_URL` in your `.env` to point to a remote Ceramic node.

### 6. Start Ceramic Service

In a new terminal:

```bash
cd ceramic-service
pnpm dev
```

The service will:
- Initialize DID authentication
- Connect to Ceramic node
- Start HTTP server on port 3001
- Print your DID (starts with `did:key:`)

You should see:
```
info: Ceramic service listening on port 3001
info: DID: did:key:z6Mk...
info: Service ready to accept requests
```

### 7. Test the Integration

#### Health Check

```bash
curl http://localhost:3001/health
```

Should return:
```json
{
  "status": "healthy",
  "service": "ceramic-service",
  "did": "did:key:z6Mk...",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Ingest Documents with Ceramic Streams

```bash
cd docling-rag-agent
python -m ingestion.ingest --documents ./documents
```

The ingestion pipeline will now:
1. Parse and chunk documents
2. Generate embeddings
3. Save to PostgreSQL
4. **Create Ceramic streams for each chunk**
5. **Pin content to IPFS** (for large chunks)
6. Record stream mappings in `ceramic_streams` table

Check the logs for:
```
INFO - Created Ceramic stream: kjzl6cwe1jw14... for chunk <uuid>
INFO - Saved Ceramic stream mapping for chunk <uuid>
INFO - Created 10 Ceramic streams for document <uuid>
```

### 8. Verify Ceramic Streams

Query the database to see created streams:

```sql
SELECT 
    c.chunk_index,
    cs.stream_id,
    cs.signer_did,
    cs.ipfs_pins,
    cs.version,
    cs.created_at
FROM ceramic_streams cs
JOIN chunks c ON cs.chunk_id = c.id
ORDER BY c.chunk_index
LIMIT 10;
```

Or use the convenient view:

```sql
SELECT * FROM chunks_with_ceramic LIMIT 10;
```

## Architecture Overview

```
┌──────────────────┐
│  Docling RAG     │
│  (Python)        │
│                  │
│  1. Parse docs   │
│  2. Chunk        │
│  3. Embed        │
│  4. Save to PG   │
└────────┬─────────┘
         │ HTTP
         ▼
┌──────────────────┐      ┌──────────────────┐
│ Ceramic Service  │─────▶│ Ceramic Network  │
│ (TypeScript)     │      │ (TileDocument)   │
│                  │      └──────────────────┘
│  - Sign w/ DID   │
│  - Create stream │               │
│  - Pin to IPFS   │               ▼
└──────────────────┘      ┌──────────────────┐
                          │  IPFS Network    │
                          │  (Pinned Data)   │
                          └──────────────────┘
```

## Data Flow

### Document Ingestion

1. **Docling** parses document → markdown + metadata
2. **Chunker** splits into semantic chunks
3. **Embedder** generates vector embeddings
4. **PostgreSQL** stores chunks with embeddings
5. **Ceramic Client** (Python) calls Ceramic service:
   - Sends chunk data via HTTP POST
6. **Ceramic Service** (TypeScript):
   - Creates signed TileDocument with agent DID
   - Pins large chunks to IPFS (if > 16KB)
   - Returns stream_id, commit_cid, ipfs_pins
7. **PostgreSQL** stores mapping in `ceramic_streams` table

### Chunk Retrieval

Chunks can be retrieved from:
- **PostgreSQL** (fast, primary query layer with PGVector)
- **Ceramic Network** (authoritative, signed, immutable history)
- **IPFS** (content-addressable, permanent storage)

## Troubleshooting

### Ceramic Service Won't Start

**Error**: `AGENT_DID_SEED_BASE64 environment variable is required`

**Solution**: Generate and set DID seed in `.env`:
```bash
openssl rand -base64 32
```

**Error**: `Failed to initialize Ceramic client`

**Solution**: Ensure Ceramic One node is running on port 5101:
```bash
cd ceramic-sdk
pnpm run c1
```

### Pinata Errors

**Error**: `Pinata configuration missing`

**Solution**: Set either `PINATA_JWT` or both `PINATA_API_KEY` + `PINATA_API_SECRET` in `.env`

Get credentials from: https://app.pinata.cloud/keys

### Ceramic Streams Not Created

**Error**: Ingestion completes but no Ceramic streams in database

**Solution**: 
1. Check `ENABLE_CERAMIC_STREAMS=true` in `.env`
2. Verify Ceramic service is running: `curl http://localhost:3001/health`
3. Check logs for HTTP errors

### Database Schema Errors

**Error**: `relation "ceramic_streams" does not exist`

**Solution**: Apply the schema:
```bash
psql $DATABASE_URL -f sql/ceramic_streams.sql
```

## Advanced Configuration

### Disable Ceramic Integration (Temporarily)

Set in `.env`:
```bash
ENABLE_CERAMIC_STREAMS=false
```

The system will work normally without Ceramic, using only PostgreSQL.

### Adjust Batch Size

For faster ingestion of many documents:
```bash
CERAMIC_BATCH_SIZE=20
```

For more stable network connections:
```bash
CERAMIC_BATCH_SIZE=5
```

### Use Local IPFS

Instead of Pinata:
```bash
IPFS_PINNING_SERVICE=local
IPFS_API_URL=http://localhost:5001
```

Requires running IPFS daemon:
```bash
ipfs daemon
```

### Production Deployment

For production use:

1. **Use KMS for DID seed** (AWS KMS, Google Cloud KMS, HashiCorp Vault)
2. **Run Ceramic service as a daemon** (systemd, Docker, Kubernetes)
3. **Monitor service health** (Prometheus, Datadog)
4. **Set up key rotation** (monthly recommended)
5. **Use managed IPFS** (Pinata, Infura, Cloudflare)

See `CERAMIC_INTEGRATION.md` for detailed production deployment guide.

## Next Steps

- Read `CERAMIC_INTEGRATION.md` for complete design documentation
- Review `ceramic-service/README.md` for API details
- Check `sql/ceramic_streams.sql` for database schema
- Explore example queries in documentation

## Support

For issues or questions:
1. Check logs in both Python and TypeScript services
2. Verify environment variables are set correctly
3. Test each component independently (Ceramic node, service, Python client)
4. Review the comprehensive documentation in `CERAMIC_INTEGRATION.md`
