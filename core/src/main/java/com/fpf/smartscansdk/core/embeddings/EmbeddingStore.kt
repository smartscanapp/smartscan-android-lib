package com.fpf.smartscansdk.core.embeddings

interface EmbeddingStore {
    val exists: Boolean
    suspend fun add(embeddings: List<StoredEmbedding>): Int
    suspend fun update(embeddings: List<StoredEmbedding>): Int
    suspend fun remove(ids: List<Long>): Int
    suspend fun get(): List<StoredEmbedding>
    fun clear()
    suspend fun save()

    suspend fun query(
        embedding: Embedding,
        topK: Int,
        threshold: Float,
        ids: Set<Long> = emptySet(),
        startDate: Long? = null,
        endDate: Long? = null,
        includeSims: Boolean = false
    ): QueryResult
}
