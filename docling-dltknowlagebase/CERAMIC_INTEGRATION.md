# Ceramic Network Integration Design

## Overview

This document describes the integration of Ceramic Network with the Docling RAG system to enable:
- **Signed, verifiable** document streams using Decentralized Identifiers (DIDs)
- **Mutable persistence** on Ceramic Network with auditable history
- **IPFS pinning** for content-addressable storage
- **Provenance tracking** of document chunks and updates

## Architecture

```
┌─────────────────┐
│   Documents     │
│  (PDF, DOCX,    │
│   MD, Audio)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Docling Parser  │
│  + Chunker      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  PostgreSQL     │◄────►│  Ceramic Service │
│  + PGVector     │      │  (Node.js/TS)    │
│                 │      └────────┬─────────┘
│  ┌───────────┐  │               │
│  │ chunks    │  │               │
│  ├───────────┤  │               │
│  │ ceramic_  │  │               ▼
│  │ streams   │  │      ┌──────────────────┐
│  └───────────┘  │      │ Ceramic Network  │
└─────────────────┘      │  (TileDocument)  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  IPFS Network    │
                         │  (Pinned Content)│
                         └──────────────────┘
```

## Data Flow

### Ingestion Pipeline

1. **Document Processing**
   - Docling parses document → markdown + metadata
   - Chunker splits into semantic chunks
   - Embeddings generated for vector search

2. **Ceramic Stream Creation**
   - Python ingestion calls Ceramic service via HTTP/subprocess
   - Ceramic service:
     - Creates signed TileDocument for chunk
     - Pins content to IPFS (optional: content can be stored in stream or as IPFS ref)
     - Returns stream_id, commit_cid, and ipfs_pins
   - PostgreSQL stores mapping in `ceramic_streams` table

3. **Database Storage**
   - Chunk saved to PostgreSQL with embedding
   - Ceramic stream mapping recorded with stream_id and CIDs

### Update Pipeline

1. **Document Update Detection**
   - Modified document re-ingested
   - Existing chunks identified by document_id + chunk_index

2. **Stream Update**
   - Ceramic service updates existing TileDocument
   - New commit created (mutable stream maintains history)
   - PostgreSQL mapping updated with new CID and version

3. **Version Tracking**
   - Ceramic maintains full commit history
   - PostgreSQL tracks latest version number
   - Old commits remain accessible via Ceramic

## Ceramic Stream Schema

### TileDocument Structure

```typescript
{
  doc_id: string,           // UUID of parent document
  chunk_id: string,         // UUID of chunk
  chunk_index: number,      // Position in document
  text_cid?: string,        // IPFS CID if text stored separately
  text?: string,            // Full text if stored in stream (small chunks)
  source: string,           // Document source path
  embeddings_metadata: {
    model: string,
    dimension: number
  },
  created_at: string,       // ISO 8601 timestamp
  updated_at: string        // ISO 8601 timestamp
}
```

### Storage Strategy

**Option 1: Small Chunks (< 16KB)**
- Store full text in Ceramic stream
- Fast retrieval, no extra IPFS lookup
- Recommended for most use cases

**Option 2: Large Chunks**
- Store text on IPFS → get CID
- Store CID + metadata in Ceramic stream
- Keeps Ceramic documents small
- Two-step retrieval: Ceramic → IPFS

## Database Schema

### ceramic_streams Table

```sql
CREATE TABLE ceramic_streams (
    id UUID PRIMARY KEY,
    chunk_id UUID REFERENCES chunks(id),
    stream_id TEXT UNIQUE NOT NULL,     -- Multibase encoded stream ID
    latest_cid TEXT NOT NULL,           -- CID of latest commit
    signer_did TEXT NOT NULL,           -- DID controller
    ipfs_pins JSONB DEFAULT '[]',       -- Pin records
    stream_metadata JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);
```

### Pin Records Format

```json
[
  {
    "cid": "bafybeig...",
    "service": "pinata",
    "pinned_at": "2024-01-15T10:30:00Z",
    "status": "pinned"
  }
]
```

## DID Management

### Agent DID Strategy

**Primary Approach: `did:key` (Ed25519)**
- Deterministic DID from 32-byte seed
- Seed stored in environment variable (base64 encoded)
- No external dependencies
- Fast and simple

**Alternative: KMS-backed DID**
- For production: use AWS KMS, Google Cloud KMS, or HashiCorp Vault
- Seed never leaves secure storage
- Supports key rotation

### Environment Variables

```bash
# Ceramic Configuration
CERAMIC_URL=http://localhost:5101
CERAMIC_SERVICE_URL=http://localhost:3001

# DID Signing Key (base64-encoded 32-byte seed)
AGENT_DID_SEED_BASE64=<base64-seed>

# IPFS Pinning Service
IPFS_PINNING_SERVICE=pinata|infura|local
PINATA_API_KEY=<key>
PINATA_API_SECRET=<secret>

# Alternative: Local IPFS
IPFS_API_URL=http://localhost:5001
```

## Ceramic Service API

### Endpoints

**POST /api/ceramic/create**
- Create new TileDocument stream
- Input: chunk data + metadata
- Output: stream_id, commit_cid, ipfs_pins

**POST /api/ceramic/update**
- Update existing TileDocument
- Input: stream_id, new content
- Output: new commit_cid, updated_at

**GET /api/ceramic/stream/:streamId**
- Retrieve stream state
- Output: full stream content + metadata

**POST /api/ceramic/verify**
- Verify stream signature
- Input: stream_id
- Output: verification result + signer DID

## Implementation Phases

### Phase 1: Design & Schema ✓
- [x] Database migration for ceramic_streams table
- [x] Define stream schema
- [x] Document environment variables

### Phase 2: Ceramic Service (TypeScript)
- [ ] Setup Node.js service with Express
- [ ] Implement DID authentication (did:key)
- [ ] Create/update TileDocument operations
- [ ] IPFS pinning integration (Pinata first, then local)
- [ ] API endpoints for Python integration

### Phase 3: Python Integration
- [ ] HTTP client to call Ceramic service
- [ ] Modify ingestion pipeline to create streams
- [ ] Update chunk storage with stream mapping
- [ ] Handle stream updates on re-ingestion

### Phase 4: Testing
- [ ] Unit tests for Ceramic service
- [ ] Integration tests for full flow
- [ ] Signature verification tests
- [ ] IPFS retrieval tests

### Phase 5: Production Readiness
- [ ] KMS integration for production DIDs
- [ ] Error handling and retry logic
- [ ] Monitoring and logging
- [ ] Rate limiting and batching
- [ ] Cost optimization

## Security Considerations

### Key Management
- ❌ Never commit seed/keys to git
- ✅ Use environment variables or KMS
- ✅ Rotate keys monthly in production
- ✅ Use different DIDs for dev/staging/prod

### Access Control
- Ceramic streams controlled by agent DID
- Only agent can update streams
- Public read access for verification

### Audit Trail
- Ceramic maintains immutable commit history
- PostgreSQL version tracking
- All updates timestamped and signed

## Cost Considerations

### Ceramic Network
- Currently free on testnet
- Mainnet: minimal cost per stream
- Consider batching for high-volume ingestion

### IPFS Pinning
- Pinata: ~$0.0001 per pin/month
- Local IPFS: storage + bandwidth costs
- Strategy: Pin critical content only

### Optimization
- Batch stream creation (10-100 at a time)
- Use local caching for frequent reads
- Pin aggregated content for large documents

## Recovery & Backup

### Stream Recovery
- Streams are permanent on Ceramic
- Can always re-sync from Ceramic to PostgreSQL
- IPFS content retrievable via CID

### Database Recovery
- PostgreSQL backup includes stream mappings
- Can rebuild from Ceramic if needed
- Embeddings must be regenerated

## Monitoring

### Key Metrics
- Streams created per minute
- Failed stream creations
- IPFS pin failures
- Ceramic node response time
- DID verification failures

### Alerts
- Ceramic service down
- Pin queue backlog
- DID key expiration warning
- Signature verification failures

## Future Enhancements

1. **ComposeDB Migration**
   - Migrate from TileDocument to ComposeDB models
   - Benefit from schema validation and GraphQL queries

2. **Multi-DID Support**
   - User-specific DIDs for multi-tenant scenarios
   - DID delegation for team access

3. **Stream Subscriptions**
   - Real-time updates when streams change
   - Webhook notifications

4. **Ceramic Indexing**
   - Index streams for faster queries
   - Alternative to PostgreSQL for some queries

## References

- [Ceramic Documentation](https://developers.ceramic.network/)
- [DID Method Specification](https://w3c-ccg.github.io/did-method-key/)
- [IPFS Documentation](https://docs.ipfs.tech/)
- [Pinata API](https://docs.pinata.cloud/)
