package com.fpf.smartscansdk.core.embeddings

interface IEmbeddingStore {
    val exists: Boolean
    suspend fun add(newStoredEmbeddings: List<StoredEmbedding>)
    suspend fun remove(ids: List<Long>)
    suspend fun get(): List<StoredEmbedding>
    fun clear()

    suspend fun query(
        embedding: FloatArray,
        topK: Int,
        threshold: Float,
        filterIds: List<Long> = emptyList()
    ): List<Long>
}
