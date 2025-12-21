package com.fpf.smartscansdk.core.embeddings

// `StoredEmbedding` represents a raw vector for a single media item, with `id` corresponding to its `MediaStoreId`.
data class StoredEmbedding(
    val id: Long,
    val date: Long,
    val embedding: FloatArray
)