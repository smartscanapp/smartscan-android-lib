package com.fpf.smartscansdk.core.embeddings

sealed interface Embedding {
    data class F32(val vector: FloatArray): Embedding
    data class QInt8(val vector: ByteArray): Embedding

}
// `StoredEmbedding` represents a raw vector for a single media item, with `id` corresponding to its `MediaStoreId`.
data class StoredEmbedding(
    val id: Long,
    val date: Long,
    val embedding: Embedding
)