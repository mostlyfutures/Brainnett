import axios from 'axios'
import { logger } from './config.js'

/**
 * IPFS pin record
 */
export interface IPFSPin {
  cid: string
  service: string
  pinned_at: string
  status: 'pinned' | 'pending' | 'failed'
}

/**
 * IPFS pinning service interface
 */
export interface IPFSPinningService {
  pinContent(content: string): Promise<IPFSPin>
  pinByCID(cid: string): Promise<IPFSPin>
  getPin(cid: string): Promise<IPFSPin | null>
}

/**
 * Pinata pinning service implementation
 */
export class PinataPinningService implements IPFSPinningService {
  private apiKey: string
  private apiSecret: string
  private jwt?: string
  private baseUrl = 'https://api.pinata.cloud'

  constructor(apiKey: string, apiSecret: string, jwt?: string) {
    this.apiKey = apiKey
    this.apiSecret = apiSecret
    this.jwt = jwt
  }

  private getHeaders() {
    if (this.jwt) {
      return {
        'Authorization': `Bearer ${this.jwt}`,
        'Content-Type': 'application/json'
      }
    }
    return {
      'pinata_api_key': this.apiKey,
      'pinata_secret_api_key': this.apiSecret,
      'Content-Type': 'application/json'
    }
  }

  async pinContent(content: string): Promise<IPFSPin> {
    try {
      const response = await axios.post(
        `${this.baseUrl}/pinning/pinJSONToIPFS`,
        { pinataContent: content },
        { headers: this.getHeaders() }
      )

      const cid = response.data.IpfsHash
      logger.info(`Content pinned to Pinata: ${cid}`)

      return {
        cid,
        service: 'pinata',
        pinned_at: new Date().toISOString(),
        status: 'pinned'
      }
    } catch (error) {
      logger.error('Failed to pin content to Pinata:', error)
      throw error
    }
  }

  async pinByCID(cid: string): Promise<IPFSPin> {
    try {
      const response = await axios.post(
        `${this.baseUrl}/pinning/pinByHash`,
        { hashToPin: cid },
        { headers: this.getHeaders() }
      )

      logger.info(`CID pinned to Pinata: ${cid}`)

      return {
        cid,
        service: 'pinata',
        pinned_at: new Date().toISOString(),
        status: 'pinned'
      }
    } catch (error) {
      logger.error(`Failed to pin CID ${cid} to Pinata:`, error)
      throw error
    }
  }

  async getPin(cid: string): Promise<IPFSPin | null> {
    try {
      const response = await axios.get(
        `${this.baseUrl}/data/pinList?hashContains=${cid}`,
        { headers: this.getHeaders() }
      )

      if (response.data.count === 0) {
        return null
      }

      const pin = response.data.rows[0]
      return {
        cid: pin.ipfs_pin_hash,
        service: 'pinata',
        pinned_at: pin.date_pinned,
        status: 'pinned'
      }
    } catch (error) {
      logger.error(`Failed to get pin status for ${cid}:`, error)
      return null
    }
  }
}

/**
 * Local IPFS pinning service implementation
 */
export class LocalIPFSPinningService implements IPFSPinningService {
  private apiUrl: string

  constructor(apiUrl: string) {
    this.apiUrl = apiUrl
  }

  async pinContent(content: string): Promise<IPFSPin> {
    try {
      // Add content to IPFS
      const formData = new FormData()
      formData.append('file', new Blob([content]))

      const response = await axios.post(
        `${this.apiUrl}/api/v0/add`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          params: { pin: 'true' }
        }
      )

      const cid = response.data.Hash
      logger.info(`Content pinned to local IPFS: ${cid}`)

      return {
        cid,
        service: 'local',
        pinned_at: new Date().toISOString(),
        status: 'pinned'
      }
    } catch (error) {
      logger.error('Failed to pin content to local IPFS:', error)
      throw error
    }
  }

  async pinByCID(cid: string): Promise<IPFSPin> {
    try {
      // Pin existing CID
      await axios.post(
        `${this.apiUrl}/api/v0/pin/add`,
        null,
        { params: { arg: cid } }
      )

      logger.info(`CID pinned to local IPFS: ${cid}`)

      return {
        cid,
        service: 'local',
        pinned_at: new Date().toISOString(),
        status: 'pinned'
      }
    } catch (error) {
      logger.error(`Failed to pin CID ${cid} to local IPFS:`, error)
      throw error
    }
  }

  async getPin(cid: string): Promise<IPFSPin | null> {
    try {
      const response = await axios.post(
        `${this.apiUrl}/api/v0/pin/ls`,
        null,
        { params: { arg: cid } }
      )

      if (response.data.Keys && response.data.Keys[cid]) {
        return {
          cid,
          service: 'local',
          pinned_at: new Date().toISOString(),
          status: 'pinned'
        }
      }

      return null
    } catch (error) {
      logger.error(`Failed to get pin status for ${cid}:`, error)
      return null
    }
  }
}

/**
 * Create IPFS pinning service based on configuration
 */
export function createPinningService(
  service: 'pinata' | 'infura' | 'local',
  config: {
    pinataApiKey?: string
    pinataApiSecret?: string
    pinataJwt?: string
    ipfsApiUrl?: string
  }
): IPFSPinningService {
  switch (service) {
    case 'pinata':
      if (!config.pinataJwt && (!config.pinataApiKey || !config.pinataApiSecret)) {
        throw new Error('Pinata configuration missing')
      }
      return new PinataPinningService(
        config.pinataApiKey || '',
        config.pinataApiSecret || '',
        config.pinataJwt
      )
    
    case 'local':
      if (!config.ipfsApiUrl) {
        throw new Error('IPFS API URL missing')
      }
      return new LocalIPFSPinningService(config.ipfsApiUrl)
    
    default:
      throw new Error(`Unsupported pinning service: ${service}`)
  }
}
