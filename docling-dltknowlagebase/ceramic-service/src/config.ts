import { CeramicClient } from '@ceramic-sdk/http-client'
import { getAuthenticatedDID } from '@didtools/key-did'
import type { DID } from 'dids'
import dotenv from 'dotenv'
import winston from 'winston'

dotenv.config()

// Configure logger
export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
})

/**
 * Configuration for the Ceramic service
 */
export interface CeramicConfig {
  ceramicUrl: string
  agentDidSeed: Uint8Array
  ipfsPinningService: 'pinata' | 'infura' | 'local'
  pinataApiKey?: string
  pinataApiSecret?: string
  pinataJwt?: string
  ipfsApiUrl?: string
}

/**
 * Load and validate configuration from environment variables
 */
export function loadConfig(): CeramicConfig {
  const ceramicUrl = process.env.CERAMIC_URL || 'http://localhost:5101'
  
  // Load DID seed from environment
  const seedBase64 = process.env.AGENT_DID_SEED_BASE64
  if (!seedBase64) {
    throw new Error('AGENT_DID_SEED_BASE64 environment variable is required')
  }
  
  let agentDidSeed: Uint8Array
  try {
    const seedBuffer = Buffer.from(seedBase64, 'base64')
    if (seedBuffer.length !== 32) {
      throw new Error(`Expected 32 bytes, got ${seedBuffer.length}`)
    }
    agentDidSeed = new Uint8Array(seedBuffer)
  } catch (error) {
    throw new Error(`Invalid AGENT_DID_SEED_BASE64: ${error instanceof Error ? error.message : 'unknown error'}`)
  }
  
  const ipfsPinningService = (process.env.IPFS_PINNING_SERVICE || 'pinata') as 'pinata' | 'infura' | 'local'
  
  const config: CeramicConfig = {
    ceramicUrl,
    agentDidSeed,
    ipfsPinningService,
    pinataApiKey: process.env.PINATA_API_KEY,
    pinataApiSecret: process.env.PINATA_API_SECRET,
    pinataJwt: process.env.PINATA_JWT,
    ipfsApiUrl: process.env.IPFS_API_URL || 'http://localhost:5001'
  }
  
  // Validate pinning service configuration
  if (config.ipfsPinningService === 'pinata') {
    if (!config.pinataJwt && (!config.pinataApiKey || !config.pinataApiSecret)) {
      throw new Error('Pinata configuration missing: requires PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET)')
    }
  }
  
  return config
}

/**
 * Initialize and authenticate DID
 */
export async function initializeDID(seed: Uint8Array): Promise<DID> {
  try {
    const did = await getAuthenticatedDID(seed)
    logger.info(`DID initialized: ${did.id}`)
    return did
  } catch (error) {
    logger.error('Failed to initialize DID:', error)
    throw error
  }
}

/**
 * Initialize Ceramic client with authenticated DID
 */
export async function initializeCeramicClient(
  ceramicUrl: string,
  did: DID
): Promise<CeramicClient> {
  try {
    const client = new CeramicClient({ url: ceramicUrl })
    logger.info(`Ceramic client initialized: ${ceramicUrl}`)
    return client
  } catch (error) {
    logger.error('Failed to initialize Ceramic client:', error)
    throw error
  }
}
