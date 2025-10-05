#!/usr/bin/env python3
"""
Test script for Ceramic integration.

This script tests the full Ceramic integration:
1. Health check of Ceramic service
2. Create test streams
3. Verify streams are created in database
4. Test batch creation
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ceramic_client import CeramicClient
from utils.db_utils import initialize_database, close_database, db_pool
from dotenv import load_dotenv

load_dotenv()


async def test_health_check(client: CeramicClient):
    """Test Ceramic service health check."""
    print("\n=== Testing Health Check ===")
    health = await client.health_check()
    print(f"Health Status: {health}")
    
    if health.get('status') == 'healthy':
        print("✓ Ceramic service is healthy")
        print(f"  DID: {health.get('did')}")
        return True
    else:
        print("✗ Ceramic service is not healthy")
        return False


async def test_create_single_stream(client: CeramicClient):
    """Test creating a single Ceramic stream."""
    print("\n=== Testing Single Stream Creation ===")
    
    test_data = {
        "doc_id": "test-doc-001",
        "chunk_id": "test-chunk-001",
        "chunk_index": 0,
        "text": "This is a test chunk for Ceramic integration testing.",
        "source": "test/example.md",
        "embeddings_metadata": {
            "model": "text-embedding-3-small",
            "dimension": 1536
        }
    }
    
    result = await client.create_stream(**test_data)
    
    if result:
        print("✓ Stream created successfully")
        print(f"  Stream ID: {result['stream_id']}")
        print(f"  Commit CID: {result['commit_cid']}")
        print(f"  Signer DID: {result['signer_did']}")
        print(f"  IPFS Pins: {len(result.get('ipfs_pins', []))}")
        return result
    else:
        print("✗ Failed to create stream")
        return None


async def test_batch_create_streams(client: CeramicClient):
    """Test batch stream creation."""
    print("\n=== Testing Batch Stream Creation ===")
    
    chunks_data = []
    for i in range(5):
        chunks_data.append({
            "doc_id": "test-doc-002",
            "chunk_id": f"test-chunk-{i:03d}",
            "chunk_index": i,
            "text": f"This is test chunk number {i} for batch creation testing.",
            "source": "test/batch-example.md",
            "embeddings_metadata": {
                "model": "text-embedding-3-small",
                "dimension": 1536
            }
        })
    
    results = await client.batch_create_streams(chunks_data)
    
    successful = [r for r in results if r is not None]
    print(f"✓ Created {len(successful)}/{len(chunks_data)} streams")
    
    for i, result in enumerate(results):
        if result:
            print(f"  Stream {i+1}: {result['stream_id']}")
    
    return results


async def test_verify_stream(client: CeramicClient, stream_id: str):
    """Test stream verification."""
    print("\n=== Testing Stream Verification ===")
    
    result = await client.verify_stream(stream_id)
    
    if result.get('valid'):
        print(f"✓ Stream signature is valid")
        print(f"  Signer DID: {result.get('signer_did')}")
        return True
    else:
        print("✗ Stream signature is invalid")
        return False


async def test_database_integration():
    """Test that streams are saved to database."""
    print("\n=== Testing Database Integration ===")
    
    await initialize_database()
    
    async with db_pool.acquire() as conn:
        # Count ceramic streams
        count = await conn.fetchval("SELECT COUNT(*) FROM ceramic_streams")
        print(f"Total Ceramic streams in database: {count}")
        
        if count > 0:
            # Show recent streams
            print("\nRecent streams:")
            results = await conn.fetch(
                """
                SELECT 
                    stream_id,
                    signer_did,
                    version,
                    created_at
                FROM ceramic_streams
                ORDER BY created_at DESC
                LIMIT 5
                """
            )
            
            for row in results:
                print(f"  - {row['stream_id'][:20]}... (v{row['version']})")
            
            print("✓ Database integration working")
        else:
            print("⚠ No streams found in database yet")
            print("  Run the ingestion pipeline to create streams")
    
    await close_database()


async def main():
    """Run all tests."""
    print("="*60)
    print("Ceramic Integration Test Suite")
    print("="*60)
    
    client = CeramicClient()
    
    # Test 1: Health check
    healthy = await test_health_check(client)
    if not healthy:
        print("\n❌ Ceramic service is not running or not configured correctly")
        print("Please start the Ceramic service first:")
        print("  cd ceramic-service")
        print("  pnpm dev")
        return
    
    # Test 2: Create single stream
    stream_result = await test_create_single_stream(client)
    
    # Test 3: Batch create streams
    batch_results = await test_batch_create_streams(client)
    
    # Test 4: Verify stream (if we created one)
    if stream_result:
        await test_verify_stream(client, stream_result['stream_id'])
    
    # Test 5: Check database integration
    await test_database_integration()
    
    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print(f"  ✓ Service health check: {'PASS' if healthy else 'FAIL'}")
    print(f"  ✓ Single stream creation: {'PASS' if stream_result else 'FAIL'}")
    print(f"  ✓ Batch stream creation: {'PASS' if batch_results else 'FAIL'}")
    
    print("\nNext steps:")
    print("  1. Run ingestion pipeline: python -m ingestion.ingest")
    print("  2. Check database: psql $DATABASE_URL -c 'SELECT * FROM chunks_with_ceramic LIMIT 10;'")
    print("  3. Query Ceramic Network for your streams")


if __name__ == "__main__":
    asyncio.run(main())
