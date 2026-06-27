package com.fpf.smartscansdk.core.embeddings

import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import kotlin.collections.map

class FileEmbeddingStore(
    private val file: File,
    private val embeddingDimension: Int,
) :
    EmbeddingStore {

    companion object {
        const val TAG = "FileEmbeddingStore"
    }

    private val codec = F32EmbeddingCodec(
        embeddingDimension = embeddingDimension,
        recordSize = (8 + 8) + embeddingDimension * 4,
        headerSize = 4
    )
    private val fileMutex = Mutex()

    private var cache: LinkedHashMap<Long, StoredEmbedding> = LinkedHashMap() // initialised in get and only updated in save
    private var idToFileOffsetIndex: MutableMap<Long, Long> = mutableMapOf() // id -> file offset

    override val exists: Boolean get() = file.exists()

    override suspend fun save(): Unit = withContext(Dispatchers.IO) {
        val embeddingsList = get()
        if(embeddingsList.isEmpty()) return@withContext
        codec.writeReplace(embeddingsList, idToFileOffsetIndex, file)
    }

    private suspend fun load(): LinkedHashMap<Long, StoredEmbedding> = withContext(Dispatchers.IO) {
        if (!file.exists()) return@withContext LinkedHashMap()
        val (embedMap, idxMap) = codec.read(file)
        idToFileOffsetIndex = idxMap
        embedMap
    }

    override suspend fun get(): List<StoredEmbedding> = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (cache.isNotEmpty()) return@withContext cache.values.toList()
            cache = load()
            cache.values.toList()
        }
    }

    suspend fun get(ids: List<Long>): List<StoredEmbedding> = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (cache.isEmpty()) cache = load()
            val storedEmbeddings = mutableListOf<StoredEmbedding>()

            for (id in ids) {
                cache[id]?.let { storedEmbeddings.add(it) }
            }
            storedEmbeddings
        }
    }

    override suspend fun add(embeddings: List<StoredEmbedding>): Int = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (embeddings.isEmpty()) return@withContext 0
            if (idToFileOffsetIndex.isEmpty()) load()


            val filteredNewEmbeddings = embeddings.filterNot { it.id in idToFileOffsetIndex }
            if (filteredNewEmbeddings.isEmpty()) return@withContext 0

            validateEmbeds(filteredNewEmbeddings)

            val added = codec.append(file,  filteredNewEmbeddings, idToFileOffsetIndex)

            // Only add items to cache if it's not empty e.g after get() call, to keep it synchronized.
            // This prevents edge cases that could result in partial cache overwriting on-disk data
            // It also prevents unnecessarily keeping embeddings in memory
            if (cache.isNotEmpty()) {
                for (embedding in filteredNewEmbeddings) {
                    cache[embedding.id] = embedding
                }
            }
            added
        }
    }

    override suspend fun update(embeddings: List<StoredEmbedding>): Int = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (embeddings.isEmpty()) return@withContext 0

            if (idToFileOffsetIndex.isEmpty()) load()

            validateEmbeds(embeddings)

            val updated = codec.update(file, embeddings, idToFileOffsetIndex)
            // Only add items to cache if it's not empty e.g after get() call, to keep it synchronized.
            // This prevents edge cases that could result in partial cache overwriting on-disk data
            // It also prevents unnecessarily keeping embeddings in memory
            if(cache.isNotEmpty()){
                for(emb in embeddings) cache[emb.id] = emb
            }
            updated
        }
    }

    override suspend fun remove(ids: List<Long>): Int = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (ids.isEmpty()) return@withContext 0
            if (cache.isEmpty()) cache = load()

            var removedCount = 0
            for (id in ids) {
                if (cache.remove(id) != null) {
                    idToFileOffsetIndex.remove(id)
                    removedCount++
                }
            }
            removedCount
        }
    }

    override suspend fun query(embedding: FloatArray, topK: Int, threshold: Float, ids: Set<Long>, startDate: Long?, endDate: Long?, includeSims: Boolean): QueryResult {
        val storedEmbeddings = get().asSequence()
            .let { seq ->
                if (ids.isNotEmpty()) seq.filter { it.id in ids } else seq
            }
            .let { seq ->
                if (startDate != null) seq.filter { it.date >= startDate } else seq
            }
            .let { seq ->
                if (endDate != null) seq.filter { it.date <= endDate } else seq
            }
            .toList()

        if (storedEmbeddings.isEmpty()) return QueryResult()

        val similarities = getSimilarities(embedding, storedEmbeddings.map { it.embedding })
        val resultIndices = getTopN(similarities, topK, threshold)

        return if (resultIndices.isEmpty()) {
            QueryResult()
        }
        else{
            val ids = resultIndices.map{idx -> storedEmbeddings[idx].id }
            val sims = if(includeSims) resultIndices.map{idx -> similarities[idx] } else null
            QueryResult(ids, sims)
        }
    }

    override fun clear(){
        cache.clear()
        idToFileOffsetIndex.clear()
    }

    private fun validateEmbeds(embeddings: List<StoredEmbedding>){
        for (embedding in embeddings) {
            if (embedding.embedding.size != embeddingDimension) {
                throw SmartScanException.InvalidEmbeddingDimension(
                    "Embedding dimension mismatch. Expected $embeddingDimension, got ${embedding.embedding.size}"
                )
            }
        }
    }
}