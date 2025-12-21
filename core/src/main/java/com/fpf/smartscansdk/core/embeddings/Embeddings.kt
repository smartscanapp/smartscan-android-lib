package com.fpf.smartscansdk.core.embeddings

// `Embedding` represents a raw vector for a single media item, with `id` corresponding to its `MediaStoreId`.
data class Embedding(
    val id: Long,
    val date: Long,
    val embeddings: FloatArray
)