# Ceramic Network Integration - Deployment Guide

This guide covers production deployment of the Ceramic integration for the Docling RAG system.

## Table of Contents

1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Security Setup](#security-setup)
3. [Deployment Options](#deployment-options)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Maintenance and Operations](#maintenance-and-operations)
6. [Troubleshooting](#troubleshooting)

## Infrastructure Requirements

### Minimum Requirements

- **Ceramic Service**: 512MB RAM, 1 vCPU
- **Ceramic One Node**: 2GB RAM, 2 vCPU
- **PostgreSQL**: 4GB RAM, 2 vCPU
- **Storage**: 50GB for PostgreSQL, 20GB for Ceramic
- **Network**: Low latency between services

### Production Recommendations

- **Ceramic Service**: 1GB RAM, 2 vCPU (for high throughput)
- **Ceramic One Node**: 4GB RAM, 4 vCPU
- **PostgreSQL**: 8GB RAM, 4 vCPU
- **Storage**: SSD with 100GB+ for PostgreSQL, 50GB+ for Ceramic
- **Network**: Private VPC, load balancer for Ceramic service

## Security Setup

### 1. DID Key Management

#### Development (Environment Variables)

```bash
# Generate secure seed
openssl rand -base64 32

# Store in .env (NEVER commit!)
echo "AGENT_DID_SEED_BASE64=$(openssl rand -base64 32)" >> .env
```

#### Production (AWS KMS Example)

```typescript
// src/config.ts
import { KMS } from '@aws-sdk/client-kms'

async function getDidSeedFromKMS(): Promise<Uint8Array> {
  const kms = new KMS({ region: 'us-east-1' })
  
  const result = await kms.decrypt({
    KeyId: process.env.KMS_KEY_ID,
    CiphertextBlob: Buffer.from(process.env.ENCRYPTED_DID_SEED, 'base64')
  })
  
  return new Uint8Array(result.Plaintext!)
}
```

#### Production (HashiCorp Vault Example)

```bash
# Store seed in Vault
vault kv put secret/ceramic-did seed=$(openssl rand -base64 32)

# Retrieve in application
export AGENT_DID_SEED_BASE64=$(vault kv get -field=seed secret/ceramic-did)
```

### 2. Network Security

```yaml
# Example security group rules (AWS)
SecurityGroupIngress:
  # Ceramic Service (internal only)
  - IpProtocol: tcp
    FromPort: 3001
    ToPort: 3001
    SourceSecurityGroupId: !Ref AppSecurityGroup
  
  # Ceramic Node (internal only)
  - IpProtocol: tcp
    FromPort: 5101
    ToPort: 5101
    SourceSecurityGroupId: !Ref CeramicServiceSecurityGroup
  
  # PostgreSQL (internal only)
  - IpProtocol: tcp
    FromPort: 5432
    ToPort: 5432
    SourceSecurityGroupId: !Ref AppSecurityGroup
```

### 3. API Authentication (Future Enhancement)

Add JWT authentication to Ceramic service:

```typescript
// src/middleware/auth.ts
import jwt from 'jsonwebtoken'

export function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1]
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' })
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    req.user = decoded
    next()
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' })
  }
}
```

## Deployment Options

### Option 1: Docker Compose (Development/Small Production)

```bash
# Generate DID seed
echo "AGENT_DID_SEED_BASE64=$(openssl rand -base64 32)" >> .env

# Add Pinata credentials
echo "PINATA_JWT=your-jwt-token" >> .env

# Start all services
docker-compose -f docker-compose.ceramic.yml up -d

# Check status
docker-compose -f docker-compose.ceramic.yml ps

# View logs
docker-compose -f docker-compose.ceramic.yml logs -f ceramic-service
```

### Option 2: Kubernetes (Production)

#### Ceramic Service Deployment

```yaml
# ceramic-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ceramic-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ceramic-service
  template:
    metadata:
      labels:
        app: ceramic-service
    spec:
      containers:
      - name: ceramic-service
        image: your-registry/ceramic-service:latest
        ports:
        - containerPort: 3001
        env:
        - name: CERAMIC_URL
          value: "http://ceramic-one:5101"
        - name: AGENT_DID_SEED_BASE64
          valueFrom:
            secretKeyRef:
              name: ceramic-secrets
              key: did-seed
        - name: PINATA_JWT
          valueFrom:
            secretKeyRef:
              name: ceramic-secrets
              key: pinata-jwt
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ceramic-service
spec:
  selector:
    app: ceramic-service
  ports:
  - protocol: TCP
    port: 3001
    targetPort: 3001
  type: ClusterIP
```

#### Create Secrets

```bash
# Create Kubernetes secrets
kubectl create secret generic ceramic-secrets \
  --from-literal=did-seed=$(openssl rand -base64 32) \
  --from-literal=pinata-jwt=your-pinata-jwt-token

# Verify
kubectl get secrets ceramic-secrets
```

#### Deploy

```bash
# Deploy services
kubectl apply -f ceramic-service-deployment.yaml
kubectl apply -f ceramic-one-deployment.yaml

# Check status
kubectl get pods -l app=ceramic-service
kubectl logs -f deployment/ceramic-service
```

### Option 3: Systemd Service (Linux Server)

```ini
# /etc/systemd/system/ceramic-service.service
[Unit]
Description=Ceramic Service for Docling RAG
After=network.target postgresql.service

[Service]
Type=simple
User=ceramic
WorkingDirectory=/opt/ceramic-service
EnvironmentFile=/etc/ceramic-service/env
ExecStart=/usr/bin/node dist/index.js
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
# Setup
sudo useradd -r -s /bin/false ceramic
sudo mkdir -p /opt/ceramic-service /etc/ceramic-service
sudo cp -r ceramic-service/* /opt/ceramic-service/
sudo cp .env /etc/ceramic-service/env
sudo chown -R ceramic:ceramic /opt/ceramic-service

# Install and start
sudo systemctl daemon-reload
sudo systemctl enable ceramic-service
sudo systemctl start ceramic-service

# Check status
sudo systemctl status ceramic-service
sudo journalctl -u ceramic-service -f
```

## Monitoring and Observability

### 1. Metrics Collection

#### Prometheus Integration

```typescript
// src/metrics.ts
import promClient from 'prom-client'

const register = new promClient.Registry()

export const streamsCreated = new promClient.Counter({
  name: 'ceramic_streams_created_total',
  help: 'Total number of Ceramic streams created',
  registers: [register]
})

export const streamCreationDuration = new promClient.Histogram({
  name: 'ceramic_stream_creation_duration_seconds',
  help: 'Duration of stream creation',
  buckets: [0.1, 0.5, 1, 2, 5],
  registers: [register]
})

export const ipfsPinFailures = new promClient.Counter({
  name: 'ceramic_ipfs_pin_failures_total',
  help: 'Total number of IPFS pin failures',
  registers: [register]
})

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType)
  res.end(await register.metrics())
})
```

### 2. Logging

#### Structured JSON Logging

Already implemented in `src/config.ts` using Winston.

#### Centralized Logging (ELK Stack)

```javascript
// Add Elasticsearch transport
import { ElasticsearchTransport } from 'winston-elasticsearch'

logger.add(new ElasticsearchTransport({
  level: 'info',
  clientOpts: {
    node: process.env.ELASTICSEARCH_URL
  },
  index: 'ceramic-service'
}))
```

### 3. Alerting

#### Example Prometheus Alerts

```yaml
# alerts/ceramic-service.yml
groups:
- name: ceramic-service
  rules:
  - alert: CeramicServiceDown
    expr: up{job="ceramic-service"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Ceramic service is down"
      description: "Ceramic service has been down for more than 5 minutes"
  
  - alert: HighIPFSPinFailureRate
    expr: rate(ceramic_ipfs_pin_failures_total[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High IPFS pin failure rate"
      description: "IPFS pin failure rate is above 10% for 5 minutes"
  
  - alert: CeramicStreamCreationSlow
    expr: histogram_quantile(0.95, rate(ceramic_stream_creation_duration_seconds_bucket[5m])) > 5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Slow stream creation"
      description: "95th percentile of stream creation duration is above 5 seconds"
```

### 4. Dashboard (Grafana)

Create dashboards to monitor:
- Stream creation rate
- Stream creation latency
- IPFS pin success/failure rate
- Service health and uptime
- DID authentication status
- Database connection pool usage

## Maintenance and Operations

### 1. Key Rotation

```bash
# Generate new DID seed
NEW_SEED=$(openssl rand -base64 32)

# Update environment (zero-downtime)
# 1. Deploy new instance with new seed
# 2. Switch traffic to new instance
# 3. Shutdown old instance

# Update secret in Kubernetes
kubectl create secret generic ceramic-secrets-new \
  --from-literal=did-seed=$NEW_SEED \
  --from-literal=pinata-jwt=$PINATA_JWT

kubectl patch deployment ceramic-service \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"ceramic-service","env":[{"name":"AGENT_DID_SEED_BASE64","valueFrom":{"secretKeyRef":{"name":"ceramic-secrets-new","key":"did-seed"}}}]}]}}}}'
```

### 2. Database Maintenance

```sql
-- Vacuum ceramic_streams table
VACUUM ANALYZE ceramic_streams;

-- Reindex for performance
REINDEX TABLE ceramic_streams;

-- Check table size
SELECT 
  pg_size_pretty(pg_total_relation_size('ceramic_streams')) AS total_size,
  pg_size_pretty(pg_relation_size('ceramic_streams')) AS table_size,
  pg_size_pretty(pg_total_relation_size('ceramic_streams') - pg_relation_size('ceramic_streams')) AS indexes_size;
```

### 3. Backup and Recovery

```bash
# Backup PostgreSQL
pg_dump $DATABASE_URL -t ceramic_streams > ceramic_streams_backup.sql

# Backup Ceramic data
tar -czf ceramic_data_backup.tar.gz /path/to/ceramic/data

# Restore
psql $DATABASE_URL < ceramic_streams_backup.sql
```

### 4. Performance Tuning

#### PostgreSQL

```sql
-- Increase work_mem for complex queries
SET work_mem = '256MB';

-- Tune connection pool
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
```

#### Ceramic Service

```typescript
// Increase batch size for high-throughput
CERAMIC_BATCH_SIZE=50

// Increase timeout for slow networks
CERAMIC_TIMEOUT=60000
```

## Troubleshooting

### Common Issues

#### 1. DID Authentication Failures

**Symptoms**: `Failed to authenticate DID` errors

**Solutions**:
- Verify `AGENT_DID_SEED_BASE64` is exactly 32 bytes when decoded
- Check seed hasn't been corrupted
- Regenerate seed if needed

#### 2. Ceramic Connection Timeouts

**Symptoms**: `Connection timeout` to Ceramic node

**Solutions**:
- Check Ceramic One is running: `curl http://localhost:5101/ceramic/liveness`
- Verify network connectivity
- Check firewall rules
- Increase timeout in configuration

#### 3. IPFS Pin Failures

**Symptoms**: `Failed to pin to IPFS` errors

**Solutions**:
- Verify Pinata credentials
- Check Pinata quota/limits
- Switch to local IPFS if Pinata is down
- Implement retry logic with exponential backoff

#### 4. High Memory Usage

**Symptoms**: OOM errors, slow performance

**Solutions**:
- Reduce `CERAMIC_BATCH_SIZE`
- Increase container memory limits
- Enable garbage collection tuning
- Monitor for memory leaks

## Cost Optimization

### 1. IPFS Pinning Costs

- **Strategy**: Only pin chunks > 16KB threshold
- **Savings**: Store small chunks directly in Ceramic
- **Pinata**: ~$20/month for 100GB
- **Alternative**: Self-hosted IPFS cluster

### 2. Ceramic Network

- **Current**: Free on testnet
- **Mainnet**: Minimal per-stream costs
- **Optimization**: Batch operations, reuse models

### 3. Database Storage

- **Vacuum regularly** to reclaim space
- **Archive old streams** after 90 days
- **Use compression** for JSONB fields

## Production Checklist

- [ ] DID seed stored in KMS/Vault
- [ ] Secrets rotation policy in place
- [ ] Monitoring and alerting configured
- [ ] Backup strategy implemented
- [ ] Load balancer configured
- [ ] Auto-scaling rules defined
- [ ] Disaster recovery plan documented
- [ ] Security audit completed
- [ ] Performance baseline established
- [ ] Runbook for common issues created

## Support and Resources

- [Ceramic Documentation](https://developers.ceramic.network/)
- [IPFS Documentation](https://docs.ipfs.tech/)
- [DID Specification](https://w3c.github.io/did-core/)
- [Pinata Docs](https://docs.pinata.cloud/)
