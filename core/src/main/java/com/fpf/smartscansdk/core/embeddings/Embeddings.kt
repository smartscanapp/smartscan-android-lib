package com.fpf.smartscansdk.core.embeddings

// `Embedding` represents a raw vector for a single media item, with `id` corresponding to its `MediaStoreId`.
data class Embedding(
    val id: Long,
    val date: Long,
    val embeddings: FloatArray
)

// `PrototypeEmbedding` represents an aggregated class-level vector used for few-shot classification, with `id` corresponding to a class identifier.
data class PrototypeEmbedding(
    val id: String,
    val date: Long,
    val embeddings: FloatArray
)
