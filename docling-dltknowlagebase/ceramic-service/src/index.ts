import { loadConfig, initializeDID, initializeCeramicClient, logger } from './config.js'
import { createPinningService } from './ipfs.js'
import { CeramicStreamService } from './ceramic.js'
import { createServer } from './server.js'

/**
 * Main entry point for Ceramic service
 */
async function main() {
  try {
    logger.info('Starting Ceramic service...')

    // Load configuration
    const config = loadConfig()
    logger.info('Configuration loaded')

    // Initialize DID
    const did = await initializeDID(config.agentDidSeed)

    // Initialize Ceramic client
    const ceramicClient = await initializeCeramicClient(config.ceramicUrl, did)

    // Initialize IPFS pinning service
    const pinningService = createPinningService(config.ipfsPinningService, {
      pinataApiKey: config.pinataApiKey,
      pinataApiSecret: config.pinataApiSecret,
      pinataJwt: config.pinataJwt,
      ipfsApiUrl: config.ipfsApiUrl
    })
    logger.info(`IPFS pinning service initialized: ${config.ipfsPinningService}`)

    // Initialize Ceramic stream service
    const ceramicService = new CeramicStreamService(
      ceramicClient,
      did,
      pinningService
    )

    // Initialize the model
    await ceramicService.initializeModel()

    // Create and start Express server
    const app = createServer(ceramicService, ceramicClient, did)
    const port = process.env.PORT || 3001

    app.listen(port, () => {
      logger.info(`Ceramic service listening on port ${port}`)
      logger.info(`DID: ${did.id}`)
      logger.info(`Ceramic URL: ${config.ceramicUrl}`)
      logger.info(`IPFS service: ${config.ipfsPinningService}`)
      logger.info('Service ready to accept requests')
    })
  } catch (error) {
    logger.error('Failed to start Ceramic service:', error)
    process.exit(1)
  }
}

// Handle shutdown gracefully
process.on('SIGINT', () => {
  logger.info('Shutting down Ceramic service...')
  process.exit(0)
})

process.on('SIGTERM', () => {
  logger.info('Shutting down Ceramic service...')
  process.exit(0)
})

// Start the service
main().catch((error) => {
  logger.error('Fatal error:', error)
  process.exit(1)
})
