import express, { Request, Response } from 'express'
import cors from 'cors'
import { logger } from './config.js'
import { CeramicStreamService, type ChunkData } from './ceramic.js'
import type { IPFSPinningService } from './ipfs.js'
import type { CeramicClient } from '@ceramic-sdk/http-client'
import type { DID } from 'dids'

/**
 * Create Express API server for Ceramic operations
 */
export function createServer(
  ceramicService: CeramicStreamService,
  client: CeramicClient,
  did: DID
) {
  const app = express()

  // Middleware
  app.use(cors())
  app.use(express.json({ limit: '10mb' }))

  // Health check
  app.get('/health', (req: Request, res: Response) => {
    res.json({
      status: 'healthy',
      service: 'ceramic-service',
      did: did.id,
      timestamp: new Date().toISOString()
    })
  })

  // Create stream for a single chunk
  app.post('/api/ceramic/create', async (req: Request, res: Response) => {
    try {
      const chunkData: ChunkData = req.body

      // Validate input
      if (!chunkData.doc_id || !chunkData.chunk_id || !chunkData.source) {
        return res.status(400).json({
          error: 'Missing required fields: doc_id, chunk_id, source'
        })
      }

      // Set timestamps if not provided
      if (!chunkData.created_at) {
        chunkData.created_at = new Date().toISOString()
      }
      if (!chunkData.updated_at) {
        chunkData.updated_at = new Date().toISOString()
      }

      const result = await ceramicService.createStream(chunkData)

      logger.info(`API: Created stream ${result.stream_id} for chunk ${chunkData.chunk_id}`)
      res.json(result)
    } catch (error) {
      logger.error('API: Failed to create stream:', error)
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Batch create streams
  app.post('/api/ceramic/batch-create', async (req: Request, res: Response) => {
    try {
      const chunks: ChunkData[] = req.body.chunks

      if (!Array.isArray(chunks) || chunks.length === 0) {
        return res.status(400).json({
          error: 'chunks array is required and must not be empty'
        })
      }

      // Set timestamps for all chunks
      const chunksWithTimestamps = chunks.map(chunk => ({
        ...chunk,
        created_at: chunk.created_at || new Date().toISOString(),
        updated_at: chunk.updated_at || new Date().toISOString()
      }))

      const results = await ceramicService.batchCreateStreams(chunksWithTimestamps)

      logger.info(`API: Batch created ${results.length} streams`)
      res.json({ results, count: results.length })
    } catch (error) {
      logger.error('API: Failed to batch create streams:', error)
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Update existing stream
  app.post('/api/ceramic/update', async (req: Request, res: Response) => {
    try {
      const { stream_id, chunk_data, current_version } = req.body

      if (!stream_id || !chunk_data) {
        return res.status(400).json({
          error: 'Missing required fields: stream_id, chunk_data'
        })
      }

      // Set updated timestamp
      chunk_data.updated_at = new Date().toISOString()

      const result = await ceramicService.updateStream(
        stream_id,
        chunk_data,
        current_version || 1
      )

      logger.info(`API: Updated stream ${stream_id}`)
      res.json(result)
    } catch (error) {
      logger.error('API: Failed to update stream:', error)
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Get stream data
  app.get('/api/ceramic/stream/:streamId', async (req: Request, res: Response) => {
    try {
      const { streamId } = req.params

      const stream = await ceramicService.getStream(streamId)

      logger.info(`API: Retrieved stream ${streamId}`)
      res.json(stream)
    } catch (error) {
      logger.error('API: Failed to get stream:', error)
      res.status(404).json({
        error: error instanceof Error ? error.message : 'Stream not found'
      })
    }
  })

  // Verify stream signature
  app.post('/api/ceramic/verify', async (req: Request, res: Response) => {
    try {
      const { stream_id } = req.body

      if (!stream_id) {
        return res.status(400).json({
          error: 'Missing required field: stream_id'
        })
      }

      const result = await ceramicService.verifyStream(stream_id)

      logger.info(`API: Verified stream ${stream_id}`)
      res.json(result)
    } catch (error) {
      logger.error('API: Failed to verify stream:', error)
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Get DID info
  app.get('/api/did', (req: Request, res: Response) => {
    res.json({
      did: did.id,
      authenticated: did.authenticated
    })
  })

  return app
}
