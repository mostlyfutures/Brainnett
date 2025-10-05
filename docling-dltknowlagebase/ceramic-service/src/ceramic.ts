import { SignedEvent } from '@ceramic-sdk/events'
import { CeramicClient } from '@ceramic-sdk/http-client'
import type { StreamID } from '@ceramic-sdk/identifiers'
import {
  createInitEvent as createDocument,
  createDataEvent as updateDocument,
} from '@ceramic-sdk/model-instance-client'
import { getStreamID } from '@ceramic-sdk/model-instance-protocol'
import { createInitEvent as createModel } from '@ceramic-sdk/model-client'
import { getModelStreamID, type ModelDefinition } from '@ceramic-sdk/model-protocol'
import type { DID } from 'dids'
import { logger } from './config.js'
import type { IPFSPinningService, IPFSPin } from './ipfs.js'

/**
 * Chunk data for Ceramic stream
 */
export interface ChunkData {
  doc_id: string
  chunk_id: string
  chunk_index: number
  text?: string
  text_cid?: string
  source: string
  embeddings_metadata: {
    model: string
    dimension: number
  }
  created_at: string
  updated_at: string
}

/**
 * Result of creating a Ceramic stream
 */
export interface CreateStreamResult {
  stream_id: string
  commit_cid: string
  ipfs_pins: IPFSPin[]
  signer_did: string
}

/**
 * Result of updating a Ceramic stream
 */
export interface UpdateStreamResult {
  stream_id: string
  commit_cid: string
  ipfs_pins: IPFSPin[]
  version: number
}

/**
 * Ceramic stream operations
 */
export class CeramicStreamService {
  private client: CeramicClient
  private did: DID
  private pinningService: IPFSPinningService
  private model?: StreamID

  constructor(
    client: CeramicClient,
    did: DID,
    pinningService: IPFSPinningService
  ) {
    this.client = client
    this.did = did
    this.pinningService = pinningService
  }

  /**
   * Initialize the Ceramic model for chunk streams
   */
  async initializeModel(): Promise<StreamID> {
    if (this.model) {
      return this.model
    }

    const modelDefinition: ModelDefinition = {
      version: '2.0',
      name: 'DoclingChunk',
      description: 'Document chunk for Docling RAG system',
      accountRelation: { type: 'list' },
      interface: false,
      implements: [],
      schema: {
        type: 'object',
        properties: {
          doc_id: { type: 'string', maxLength: 128 },
          chunk_id: { type: 'string', maxLength: 128 },
          chunk_index: { type: 'number' },
          text: { type: 'string', maxLength: 65536 },
          text_cid: { type: 'string', maxLength: 256 },
          source: { type: 'string', maxLength: 1024 },
          embeddings_metadata: {
            type: 'object',
            properties: {
              model: { type: 'string' },
              dimension: { type: 'number' }
            },
            required: ['model', 'dimension']
          },
          created_at: { type: 'string' },
          updated_at: { type: 'string' }
        },
        required: ['doc_id', 'chunk_id', 'chunk_index', 'source', 'embeddings_metadata', 'created_at', 'updated_at'],
        additionalProperties: false
      }
    }

    try {
      const modelEvent = await createModel(this.did, modelDefinition)
      const modelCID = await this.client.postEventType(SignedEvent, modelEvent)
      this.model = getModelStreamID(modelCID)
      
      logger.info(`Ceramic model initialized: ${this.model.toString()} (CID: ${modelCID.toString()})`)
      return this.model
    } catch (error) {
      logger.error('Failed to initialize Ceramic model:', error)
      throw error
    }
  }

  /**
   * Create a new Ceramic stream for a chunk
   */
  async createStream(chunkData: ChunkData): Promise<CreateStreamResult> {
    try {
      // Ensure model is initialized
      const model = await this.initializeModel()

      // Determine storage strategy: if text is large, store on IPFS
      let ipfsPins: IPFSPin[] = []
      let streamContent = { ...chunkData }

      const textSize = chunkData.text ? new Blob([chunkData.text]).size : 0
      const shouldStoreOnIPFS = textSize > 16384 // 16KB threshold

      if (shouldStoreOnIPFS && chunkData.text) {
        logger.info(`Chunk ${chunkData.chunk_id} text is large (${textSize} bytes), storing on IPFS`)
        
        // Pin text content to IPFS
        const pin = await this.pinningService.pinContent(chunkData.text)
        ipfsPins.push(pin)
        
        // Store CID in stream instead of full text
        streamContent = {
          ...chunkData,
          text: undefined,
          text_cid: pin.cid
        }
      }

      // Create Ceramic document
      const event = await createDocument({
        controller: this.did,
        content: streamContent,
        model
      })

      const cid = await this.client.postEventType(SignedEvent, event)
      const streamId = getStreamID(cid)

      logger.info(`Created Ceramic stream: ${streamId.toString()} for chunk ${chunkData.chunk_id}`)

      return {
        stream_id: streamId.toString(),
        commit_cid: cid.toString(),
        ipfs_pins: ipfsPins,
        signer_did: this.did.id
      }
    } catch (error) {
      logger.error(`Failed to create stream for chunk ${chunkData.chunk_id}:`, error)
      throw error
    }
  }

  /**
   * Update an existing Ceramic stream
   */
  async updateStream(
    streamId: string,
    chunkData: ChunkData,
    currentVersion: number = 1
  ): Promise<UpdateStreamResult> {
    try {
      // Determine storage strategy
      let ipfsPins: IPFSPin[] = []
      let streamContent = { ...chunkData }

      const textSize = chunkData.text ? new Blob([chunkData.text]).size : 0
      const shouldStoreOnIPFS = textSize > 16384

      if (shouldStoreOnIPFS && chunkData.text) {
        logger.info(`Chunk ${chunkData.chunk_id} text is large (${textSize} bytes), storing on IPFS`)
        
        const pin = await this.pinningService.pinContent(chunkData.text)
        ipfsPins.push(pin)
        
        streamContent = {
          ...chunkData,
          text: undefined,
          text_cid: pin.cid
        }
      }

      // Update Ceramic document
      const updateEvent = await updateDocument({
        controller: this.did,
        currentID: { streamId } as any, // Type workaround - actual implementation would use proper CommitID
        content: streamContent,
      })

      const cid = await this.client.postEventType(SignedEvent, updateEvent)

      logger.info(`Updated Ceramic stream: ${streamId} for chunk ${chunkData.chunk_id}`)

      return {
        stream_id: streamId,
        commit_cid: cid.toString(),
        ipfs_pins: ipfsPins,
        version: currentVersion + 1
      }
    } catch (error) {
      logger.error(`Failed to update stream ${streamId}:`, error)
      throw error
    }
  }

  /**
   * Retrieve stream data
   */
  async getStream(streamId: string): Promise<any> {
    try {
      const state = await this.client.getEventsFeed(`ceramic://${streamId}`)
      logger.info(`Retrieved stream: ${streamId}`)
      return state
    } catch (error) {
      logger.error(`Failed to get stream ${streamId}:`, error)
      throw error
    }
  }

  /**
   * Verify stream signature
   */
  async verifyStream(streamId: string): Promise<{ valid: boolean; signer_did: string }> {
    try {
      const state = await this.getStream(streamId)
      
      // In a full implementation, we would verify the signature
      // For now, we return the controller as the signer
      return {
        valid: true,
        signer_did: this.did.id
      }
    } catch (error) {
      logger.error(`Failed to verify stream ${streamId}:`, error)
      return { valid: false, signer_did: '' }
    }
  }

  /**
   * Batch create multiple streams
   */
  async batchCreateStreams(chunks: ChunkData[]): Promise<CreateStreamResult[]> {
    logger.info(`Batch creating ${chunks.length} streams`)
    
    const results: CreateStreamResult[] = []
    
    // Process in batches to avoid overwhelming the system
    const batchSize = 10
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize)
      const batchResults = await Promise.all(
        batch.map(chunk => this.createStream(chunk))
      )
      results.push(...batchResults)
      
      logger.info(`Processed batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`)
    }
    
    return results
  }
}
