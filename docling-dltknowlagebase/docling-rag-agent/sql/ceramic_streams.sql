-- Ceramic Streams Mapping Table
-- Tracks the relationship between document chunks and their Ceramic streams

CREATE TABLE IF NOT EXISTS ceramic_streams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Reference to the chunk this stream represents
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Ceramic stream identifier (multibase encoded)
    stream_id TEXT NOT NULL UNIQUE,
    
    -- Latest commit CID for this stream
    latest_cid TEXT NOT NULL,
    
    -- DID of the signer/controller
    signer_did TEXT NOT NULL,
    
    -- IPFS CIDs where content is pinned (JSON array)
    ipfs_pins JSONB DEFAULT '[]',
    
    -- Ceramic stream metadata
    stream_metadata JSONB DEFAULT '{}',
    
    -- Version tracking
    version INTEGER DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_ceramic_streams_chunk_id ON ceramic_streams (chunk_id);
CREATE INDEX IF NOT EXISTS idx_ceramic_streams_stream_id ON ceramic_streams (stream_id);
CREATE INDEX IF NOT EXISTS idx_ceramic_streams_signer_did ON ceramic_streams (signer_did);
CREATE INDEX IF NOT EXISTS idx_ceramic_streams_updated ON ceramic_streams (last_updated DESC);

-- Trigger to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_ceramic_streams_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_ceramic_streams_updated_at 
    BEFORE UPDATE ON ceramic_streams
    FOR EACH ROW 
    EXECUTE FUNCTION update_ceramic_streams_updated_at();

-- View for easy querying of chunks with their Ceramic streams
CREATE OR REPLACE VIEW chunks_with_ceramic AS
SELECT 
    c.id AS chunk_id,
    c.document_id,
    c.content,
    c.chunk_index,
    c.metadata AS chunk_metadata,
    d.title AS document_title,
    d.source AS document_source,
    cs.stream_id AS ceramic_stream_id,
    cs.latest_cid AS ceramic_latest_cid,
    cs.signer_did,
    cs.ipfs_pins,
    cs.version AS ceramic_version,
    cs.last_updated AS ceramic_last_updated
FROM chunks c
LEFT JOIN ceramic_streams cs ON c.id = cs.chunk_id
LEFT JOIN documents d ON c.document_id = d.document_id;

COMMENT ON TABLE ceramic_streams IS 'Maps document chunks to their corresponding Ceramic streams and IPFS pins';
COMMENT ON COLUMN ceramic_streams.stream_id IS 'Ceramic stream identifier (multibase encoded string)';
COMMENT ON COLUMN ceramic_streams.latest_cid IS 'CID of the latest commit to this stream';
COMMENT ON COLUMN ceramic_streams.signer_did IS 'DID that signed and controls this stream';
COMMENT ON COLUMN ceramic_streams.ipfs_pins IS 'JSON array of IPFS pin records with CID, timestamp, and service info';
COMMENT ON COLUMN ceramic_streams.version IS 'Version counter, incremented on each update';
