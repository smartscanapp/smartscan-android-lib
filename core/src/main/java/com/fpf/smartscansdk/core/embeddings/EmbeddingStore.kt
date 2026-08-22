package com.fpf.smartscansdk.core.embeddings

interface EmbeddingStore {
    val exists: Boolean
    suspend fun add(embeddings: List<StoredEmbedding>): Int

    suspend fun add(embedding: StoredEmbedding): Int = add(listOf(embedding))

    suspend fun update(embeddings: List<StoredEmbedding>): Int
    suspend fun update(embedding: StoredEmbedding): Int = update(listOf(embedding))

    suspend fun remove(ids: List<Long>): Int
    suspend fun remove(id: Long): Int = remove(listOf(id))
    suspend fun get(): List<StoredEmbedding>
    suspend fun get(ids: List<Long>): List<StoredEmbedding>

    suspend fun get(id: Long): StoredEmbedding? = get(listOf(id)).firstOrNull()

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
