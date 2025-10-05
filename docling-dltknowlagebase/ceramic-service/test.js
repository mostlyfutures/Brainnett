#!/usr/bin/env node
/**
 * Test script for Ceramic service
 * 
 * Tests the core functionality of the Ceramic service:
 * - DID initialization
 * - Model creation
 * - Stream creation
 * - Stream updates
 */

import { loadConfig, initializeDID, initializeCeramicClient, logger } from './src/config.js'
import { createPinningService } from './src/ipfs.js'
import { CeramicStreamService } from './src/ceramic.js'

async function testDIDInit() {
  console.log('\n=== Testing DID Initialization ===')
  
  const config = loadConfig()
  const did = await initializeDID(config.agentDidSeed)
  
  console.log('✓ DID initialized successfully')
  console.log(`  DID: ${did.id}`)
  console.log(`  Authenticated: ${did.authenticated}`)
  
  return did
}

async function testCeramicClient(config, did) {
  console.log('\n=== Testing Ceramic Client ===')
  
  const client = await initializeCeramicClient(config.ceramicUrl, did)
  
  console.log('✓ Ceramic client initialized')
  console.log(`  URL: ${config.ceramicUrl}`)
  
  return client
}

async function testModelCreation(service) {
  console.log('\n=== Testing Model Creation ===')
  
  const model = await service.initializeModel()
  
  console.log('✓ Model created successfully')
  console.log(`  Model ID: ${model.toString()}`)
  
  return model
}

async function testStreamCreation(service) {
  console.log('\n=== Testing Stream Creation ===')
  
  const testChunk = {
    doc_id: 'test-doc-001',
    chunk_id: 'test-chunk-001',
    chunk_index: 0,
    text: 'This is a test chunk for Ceramic service testing. It contains some sample text to verify that stream creation works correctly.',
    source: 'test/example.md',
    embeddings_metadata: {
      model: 'text-embedding-3-small',
      dimension: 1536
    },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
  
  const result = await service.createStream(testChunk)
  
  console.log('✓ Stream created successfully')
  console.log(`  Stream ID: ${result.stream_id}`)
  console.log(`  Commit CID: ${result.commit_cid}`)
  console.log(`  Signer DID: ${result.signer_did}`)
  console.log(`  IPFS Pins: ${result.ipfs_pins.length}`)
  
  return result
}

async function testBatchCreation(service) {
  console.log('\n=== Testing Batch Stream Creation ===')
  
  const testChunks = []
  for (let i = 0; i < 3; i++) {
    testChunks.push({
      doc_id: 'test-doc-002',
      chunk_id: `test-chunk-${i.toString().padStart(3, '0')}`,
      chunk_index: i,
      text: `This is test chunk number ${i} for batch creation testing.`,
      source: 'test/batch-example.md',
      embeddings_metadata: {
        model: 'text-embedding-3-small',
        dimension: 1536
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    })
  }
  
  const results = await service.batchCreateStreams(testChunks)
  
  console.log(`✓ Batch created ${results.length} streams`)
  results.forEach((result, i) => {
    console.log(`  Stream ${i + 1}: ${result.stream_id}`)
  })
  
  return results
}

async function testStreamRetrieval(service, streamId) {
  console.log('\n=== Testing Stream Retrieval ===')
  
  const stream = await service.getStream(streamId)
  
  console.log('✓ Stream retrieved successfully')
  console.log(`  Stream ID: ${streamId}`)
  
  return stream
}

async function testStreamVerification(service, streamId) {
  console.log('\n=== Testing Stream Verification ===')
  
  const result = await service.verifyStream(streamId)
  
  console.log(`✓ Stream verification: ${result.valid ? 'VALID' : 'INVALID'}`)
  console.log(`  Signer DID: ${result.signer_did}`)
  
  return result
}

async function main() {
  console.log('='​.repeat(60))
  console.log('Ceramic Service Test Suite')
  console.log('='​.repeat(60))
  
  try {
    // Load configuration
    console.log('\n=== Loading Configuration ===')
    const config = loadConfig()
    console.log('✓ Configuration loaded')
    console.log(`  Ceramic URL: ${config.ceramicUrl}`)
    console.log(`  IPFS Service: ${config.ipfsPinningService}`)
    
    // Test DID
    const did = await testDIDInit()
    
    // Test Ceramic client
    const client = await testCeramicClient(config, did)
    
    // Initialize pinning service
    console.log('\n=== Initializing IPFS Pinning ===')
    const pinningService = createPinningService(config.ipfsPinningService, {
      pinataApiKey: config.pinataApiKey,
      pinataApiSecret: config.pinataApiSecret,
      pinataJwt: config.pinataJwt,
      ipfsApiUrl: config.ipfsApiUrl
    })
    console.log(`✓ IPFS pinning service initialized: ${config.ipfsPinningService}`)
    
    // Create service
    const service = new CeramicStreamService(client, did, pinningService)
    
    // Test model creation
    await testModelCreation(service)
    
    // Test stream creation
    const streamResult = await testStreamCreation(service)
    
    // Test batch creation
    await testBatchCreation(service)
    
    // Test stream retrieval
    await testStreamRetrieval(service, streamResult.stream_id)
    
    // Test verification
    await testStreamVerification(service, streamResult.stream_id)
    
    console.log('\n' + '='​.repeat(60))
    console.log('All Tests Passed! ✓')
    console.log('='​.repeat(60))
    
    console.log('\nNext steps:')
    console.log('  1. Start the service: pnpm dev')
    console.log('  2. Test the API: curl http://localhost:3001/health')
    console.log('  3. Integrate with Python: python test_ceramic_integration.py')
    
    process.exit(0)
  } catch (error) {
    console.error('\n❌ Test failed:', error)
    process.exit(1)
  }
}

main()
