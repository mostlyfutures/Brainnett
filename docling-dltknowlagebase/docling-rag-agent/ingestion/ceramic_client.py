"""
Ceramic Network client for Python integration.

This module provides a Python client to interact with the Ceramic service
for creating and managing signed document streams on Ceramic Network.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CeramicClient:
    """Client for interacting with the Ceramic service API."""
    
    def __init__(
        self,
        service_url: Optional[str] = None,
        enabled: bool = True,
        batch_size: int = 10,
        retry_attempts: int = 3
    ):
        """
        Initialize Ceramic client.
        
        Args:
            service_url: URL of the Ceramic service (default: from env CERAMIC_SERVICE_URL)
            enabled: Whether Ceramic integration is enabled (default: from env ENABLE_CERAMIC_STREAMS)
            batch_size: Number of streams to create in a single batch (default: from env CERAMIC_BATCH_SIZE)
            retry_attempts: Number of retry attempts for failed requests (default: from env CERAMIC_RETRY_ATTEMPTS)
        """
        self.service_url = service_url or os.getenv('CERAMIC_SERVICE_URL', 'http://localhost:3001')
        self.enabled = enabled if enabled is not None else os.getenv('ENABLE_CERAMIC_STREAMS', 'true').lower() == 'true'
        self.batch_size = batch_size or int(os.getenv('CERAMIC_BATCH_SIZE', '10'))
        self.retry_attempts = retry_attempts or int(os.getenv('CERAMIC_RETRY_ATTEMPTS', '3'))
        
        if not self.enabled:
            logger.info("Ceramic integration is disabled")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the Ceramic service is healthy.
        
        Returns:
            Service health status
        """
        if not self.enabled:
            return {"status": "disabled"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.service_url}/health") as response:
                    return await response.json()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {"status": "error", "error": str(e)}
    
    async def create_stream(
        self,
        doc_id: str,
        chunk_id: str,
        chunk_index: int,
        text: str,
        source: str,
        embeddings_metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a Ceramic stream for a single chunk.
        
        Args:
            doc_id: Document ID
            chunk_id: Chunk ID
            chunk_index: Chunk index in document
            text: Chunk text content
            source: Document source path
            embeddings_metadata: Metadata about embeddings (model, dimension)
        
        Returns:
            Stream creation result with stream_id, commit_cid, ipfs_pins, signer_did
        """
        if not self.enabled:
            return None
        
        chunk_data = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "text": text,
            "source": source,
            "embeddings_metadata": embeddings_metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.retry_attempts):
                try:
                    async with session.post(
                        f"{self.service_url}/api/ceramic/create",
                        json=chunk_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Created Ceramic stream: {result['stream_id']} for chunk {chunk_id}")
                            return result
                        else:
                            error = await response.text()
                            logger.error(f"Failed to create stream (status {response.status}): {error}")
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{self.retry_attempts} failed: {e}")
                    if attempt == self.retry_attempts - 1:
                        logger.error(f"Failed to create stream for chunk {chunk_id} after {self.retry_attempts} attempts")
                        return None
        
        return None
    
    async def batch_create_streams(
        self,
        chunks_data: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Create multiple Ceramic streams in batches.
        
        Args:
            chunks_data: List of chunk data dictionaries
        
        Returns:
            List of stream creation results
        """
        if not self.enabled:
            return [None] * len(chunks_data)
        
        results = []
        
        # Process in batches
        for i in range(0, len(chunks_data), self.batch_size):
            batch = chunks_data[i:i + self.batch_size]
            
            # Ensure all chunks have timestamps
            for chunk in batch:
                if 'created_at' not in chunk:
                    chunk['created_at'] = datetime.now().isoformat()
                if 'updated_at' not in chunk:
                    chunk['updated_at'] = datetime.now().isoformat()
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(self.retry_attempts):
                    try:
                        async with session.post(
                            f"{self.service_url}/api/ceramic/batch-create",
                            json={"chunks": batch},
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                batch_results = result.get('results', [])
                                results.extend(batch_results)
                                logger.info(f"Batch created {len(batch_results)} streams")
                                break
                            else:
                                error = await response.text()
                                logger.error(f"Batch create failed (status {response.status}): {error}")
                    except Exception as e:
                        logger.error(f"Batch attempt {attempt + 1}/{self.retry_attempts} failed: {e}")
                        if attempt == self.retry_attempts - 1:
                            logger.error(f"Failed to batch create streams after {self.retry_attempts} attempts")
                            # Add None for failed chunks
                            results.extend([None] * len(batch))
        
        return results
    
    async def update_stream(
        self,
        stream_id: str,
        chunk_data: Dict[str, Any],
        current_version: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing Ceramic stream.
        
        Args:
            stream_id: Ceramic stream ID
            chunk_data: Updated chunk data
            current_version: Current stream version
        
        Returns:
            Update result with new commit_cid and version
        """
        if not self.enabled:
            return None
        
        chunk_data['updated_at'] = datetime.now().isoformat()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.retry_attempts):
                try:
                    async with session.post(
                        f"{self.service_url}/api/ceramic/update",
                        json={
                            "stream_id": stream_id,
                            "chunk_data": chunk_data,
                            "current_version": current_version
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Updated Ceramic stream: {stream_id}")
                            return result
                        else:
                            error = await response.text()
                            logger.error(f"Failed to update stream (status {response.status}): {error}")
                except Exception as e:
                    logger.error(f"Update attempt {attempt + 1}/{self.retry_attempts} failed: {e}")
                    if attempt == self.retry_attempts - 1:
                        logger.error(f"Failed to update stream {stream_id} after {self.retry_attempts} attempts")
                        return None
        
        return None
    
    async def get_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stream data from Ceramic.
        
        Args:
            stream_id: Ceramic stream ID
        
        Returns:
            Stream data
        """
        if not self.enabled:
            return None
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.service_url}/api/ceramic/stream/{stream_id}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get stream {stream_id}: status {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Failed to get stream {stream_id}: {e}")
                return None
    
    async def verify_stream(self, stream_id: str) -> Dict[str, Any]:
        """
        Verify stream signature.
        
        Args:
            stream_id: Ceramic stream ID
        
        Returns:
            Verification result with valid flag and signer_did
        """
        if not self.enabled:
            return {"valid": False, "signer_did": ""}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.service_url}/api/ceramic/verify",
                    json={"stream_id": stream_id},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"valid": False, "signer_did": ""}
            except Exception as e:
                logger.error(f"Failed to verify stream {stream_id}: {e}")
                return {"valid": False, "signer_did": ""}


# Convenience functions for direct use
async def create_ceramic_stream(
    doc_id: str,
    chunk_id: str,
    chunk_index: int,
    text: str,
    source: str,
    embeddings_metadata: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Convenience function to create a single stream."""
    client = CeramicClient()
    return await client.create_stream(
        doc_id, chunk_id, chunk_index, text, source, embeddings_metadata
    )


async def batch_create_ceramic_streams(
    chunks_data: List[Dict[str, Any]]
) -> List[Optional[Dict[str, Any]]]:
    """Convenience function to batch create streams."""
    client = CeramicClient()
    return await client.batch_create_streams(chunks_data)
