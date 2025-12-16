package com.fpf.smartscansdk.core.embeddings

interface IEmbeddingStore {
    val exists: Boolean
    suspend fun add(newEmbeddings: List<Embedding>)
    suspend fun remove(ids: List<Long>)
    suspend fun get(): List<Embedding>
    fun clear()

    suspend fun query(
        embedding: FloatArray,
        topK: Int,
        threshold: Float
    ): List<Long>
}
