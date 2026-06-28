package com.fpf.smartscansdk.core.embeddings

sealed interface Embedding {
    val size: Int
        get() = when(this){
            is F32 -> this.vector.size
            is QInt8 -> this.vector.size
        }
    data class F32(val vector: FloatArray): Embedding
    data class QInt8(val vector: ByteArray): Embedding

}

data class StoredEmbedding(
    val id: Long,
    val date: Long,
    val embedding: Embedding
)