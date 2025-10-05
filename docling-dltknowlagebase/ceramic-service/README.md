# Ceramic Service for Docling RAG

A TypeScript/Node.js service that integrates Ceramic Network with the Docling RAG system. This service handles creating signed, mutable document streams and pinning content to IPFS.

## Features

- **Signed Ceramic Streams**: Create and update TileDocument streams signed with DID
- **IPFS Pinning**: Pin content to Pinata or local IPFS nodes
- **Smart Storage**: Automatically stores large chunks on IPFS and metadata on Ceramic
- **Batch Operations**: Efficiently create multiple streams in batches
- **RESTful API**: Easy integration with Python ingestion pipeline

## Prerequisites

- Node.js 18+ (for TypeScript ES modules)
- Running Ceramic One node (default: http://localhost:5101)
- IPFS pinning service (Pinata account or local IPFS node)

## Installation

```bash
# Install dependencies (from ceramic-sdk root)
cd ../ceramic-sdk
pnpm install

# Install ceramic-service dependencies
cd ../ceramic-service
pnpm install

# Build the service
pnpm build
```

## Configuration

Create a `.env` file in the ceramic-service directory:

```bash
# Ceramic Configuration
CERAMIC_URL=http://localhost:5101

# DID Signing Key (generate with: openssl rand -base64 32)
AGENT_DID_SEED_BASE64=your-base64-encoded-seed

# IPFS Pinning Service
IPFS_PINNING_SERVICE=pinata

# Pinata Configuration
PINATA_JWT=your-pinata-jwt
# OR use API key/secret
PINATA_API_KEY=your-api-key
PINATA_API_SECRET=your-api-secret

# Local IPFS (alternative to Pinata)
# IPFS_API_URL=http://localhost:5001

# Server Configuration
PORT=3001
LOG_LEVEL=info
```

### Generate DID Seed

```bash
openssl rand -base64 32
```

Copy the output to `AGENT_DID_SEED_BASE64` in your `.env` file.

## Running the Service

### Development Mode (with auto-reload)

```bash
pnpm dev
```

### Production Mode

```bash
pnpm build
pnpm start
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status and DID information.

### Create Stream

```bash
POST /api/ceramic/create
Content-Type: application/json

{
  "doc_id": "uuid-of-document",
  "chunk_id": "uuid-of-chunk",
  "chunk_index": 0,
  "text": "Full chunk text content",
  "source": "documents/example.pdf",
  "embeddings_metadata": {
    "model": "text-embedding-3-small",
    "dimension": 1536
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

Returns:
```json
{
  "stream_id": "kjzl6cwe1jw14...",
  "commit_cid": "bafyrei...",
  "ipfs_pins": [],
  "signer_did": "did:key:z6Mk..."
}
```

### Batch Create Streams

```bash
POST /api/ceramic/batch-create
Content-Type: application/json

{
  "chunks": [
    { /* chunk data 1 */ },
    { /* chunk data 2 */ }
  ]
}
```

### Update Stream

```bash
POST /api/ceramic/update
Content-Type: application/json

{
  "stream_id": "kjzl6cwe1jw14...",
  "chunk_data": { /* updated chunk data */ },
  "current_version": 1
}
```

### Get Stream

```bash
GET /api/ceramic/stream/:streamId
```

### Verify Stream

```bash
POST /api/ceramic/verify
Content-Type: application/json

{
  "stream_id": "kjzl6cwe1jw14..."
}
```

### Get DID

```bash
GET /api/did
```

## Integration with Python

See the Python client in `docling-rag-agent/ingestion/ceramic_client.py` for integration examples.

Example:
```python
import requests

# Create stream
response = requests.post(
    'http://localhost:3001/api/ceramic/create',
    json={
        'doc_id': doc_id,
        'chunk_id': chunk_id,
        'chunk_index': 0,
        'text': 'chunk content',
        'source': 'documents/example.pdf',
        'embeddings_metadata': {
            'model': 'text-embedding-3-small',
            'dimension': 1536
        }
    }
)

result = response.json()
stream_id = result['stream_id']
```

## Storage Strategy

- **Small chunks (< 16KB)**: Full text stored in Ceramic stream
- **Large chunks (>= 16KB)**: Text pinned to IPFS, CID stored in Ceramic stream

This optimizes for both performance and cost.

## Security

- **DID Key Management**: Store `AGENT_DID_SEED_BASE64` securely
- **Never commit secrets**: Add `.env` to `.gitignore`
- **Production**: Consider using KMS (AWS/Google Cloud) for key storage
- **Key Rotation**: Implement monthly key rotation in production

## Monitoring

The service logs all operations to stdout in JSON format. Key metrics:

- Stream creation/update success/failure rates
- IPFS pin success/failure rates
- API response times
- DID authentication status

## Development

### Project Structure

```
ceramic-service/
├── src/
│   ├── index.ts        # Main entry point
│   ├── server.ts       # Express API server
│   ├── ceramic.ts      # Ceramic stream operations
│   ├── ipfs.ts         # IPFS pinning services
│   └── config.ts       # Configuration and DID setup
├── package.json
├── tsconfig.json
└── README.md
```

### Building

```bash
pnpm build
```

Compiles TypeScript to JavaScript in the `dist/` directory.

## Troubleshooting

### Ceramic Connection Issues

- Ensure Ceramic One node is running: `http://localhost:5101`
- Check Ceramic SDK is installed: `cd ../ceramic-sdk && pnpm install`

### DID Authentication Errors

- Verify `AGENT_DID_SEED_BASE64` is exactly 32 bytes when decoded
- Generate new seed: `openssl rand -base64 32`

### IPFS Pinning Failures

- **Pinata**: Verify API credentials are correct
- **Local IPFS**: Ensure IPFS daemon is running

### Port Already in Use

Change the `PORT` in `.env` or set environment variable:
```bash
PORT=3002 pnpm start
```

## License

MIT
