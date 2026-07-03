CREATE TABLE IF NOT EXISTS channels (
    youtube_channel_id TEXT PRIMARY KEY,
    channel_handle TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_channels_channel_handle
ON channels (channel_handle)
WHERE channel_handle IS NOT NULL;

CREATE TABLE IF NOT EXISTS videos (
    youtube_video_id TEXT PRIMARY KEY,
    channel_id TEXT,
    title TEXT,
    upload_date TEXT,
    transcript TEXT,
    summary TEXT,
    embedding_content_hash TEXT,
    embedding F32_BLOB,
    FOREIGN KEY (channel_id) REFERENCES channels (youtube_channel_id)
);

CREATE INDEX IF NOT EXISTS idx_videos_channel_id
ON videos (channel_id);

CREATE TABLE IF NOT EXISTS channels_watchlist (
    channel_id TEXT PRIMARY KEY,
    FOREIGN KEY (channel_id) REFERENCES channels (youtube_channel_id)
);

CREATE TABLE IF NOT EXISTS watchlist_videos (
    video_id TEXT PRIMARY KEY,
    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos (youtube_video_id)
);

CREATE TABLE IF NOT EXISTS channels_blacklist (
    channel_id TEXT PRIMARY KEY,
    FOREIGN KEY (channel_id) REFERENCES channels (youtube_channel_id)
);
