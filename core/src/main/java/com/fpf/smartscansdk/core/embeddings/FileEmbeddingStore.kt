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
    private val quantize: Boolean = false
) :
    EmbeddingStore {

    companion object {
        const val TAG = "FileEmbeddingStore"
        private const val HEADER_SIZE = 4
    }

    private val codec = if(quantize){
        QInt8EmbeddingCodec(
            embeddingDimension = embeddingDimension,
            headerSize = HEADER_SIZE
        )
    }else{
        F32EmbeddingCodec(
            embeddingDimension = embeddingDimension,
            headerSize = HEADER_SIZE
        )
    }
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
        try {
            val (embedMap, idxMap) = codec.read(file)
            idToFileOffsetIndex = idxMap
            embedMap
        }catch (e: SmartScanException.InvalidEmbeddingStoreFile){
            throw handleInvalidEmbedStoreError(e)
        }
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

            validateEmbeds(filteredNewEmbeddings, quantize)

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

            validateEmbeds(embeddings, quantize)

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

    override suspend fun query(embedding: Embedding, topK: Int, threshold: Float, ids: Set<Long>, startDate: Long?, endDate: Long?, includeSims: Boolean): QueryResult {
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

        validateQuery(quantize, embedding, storedEmbeddings[0])

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

    private fun validateEmbeds(embeddings: List<StoredEmbedding>, quantize: Boolean) {
        val isQuantEmbed = embeddings[0].embedding is Embedding.QInt8

        for (embedding in embeddings) {
            val size = when (val e = embedding.embedding) {
                is Embedding.F32 -> e.vector.size
                is Embedding.QInt8 -> e.vector.size
            }
            when{
                size != embeddingDimension -> throw SmartScanException.InvalidEmbeddingDimension("Embedding dimension mismatch. Expected $embeddingDimension, got $size")
                quantize && !isQuantEmbed -> throw SmartScanException.InvalidEmbeddingType("Embedding must be of type QInt8")
                !quantize && isQuantEmbed -> throw SmartScanException.InvalidEmbeddingType("Embedding must be of type F32")
            }
        }
    }

    private fun validateQuery(quantize: Boolean, queryEmbed: Embedding, storedEmbed: StoredEmbedding) {
        val size = queryEmbed.size
        if(size != embeddingDimension) throw SmartScanException.InvalidEmbeddingDimension("Embedding dimension mismatch. Expected $embeddingDimension, got $size")
        if (quantize){
            if(queryEmbed !is Embedding.QInt8) throw SmartScanException.InvalidEmbeddingType("Embedding must be of type QInt8")
            if(storedEmbed.embedding !is Embedding.QInt8) throw SmartScanException.InvalidEmbeddingType("Mismatch between query embedding and stored embeddings. Both must be of type QInt8")
        }else{
            if(queryEmbed !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Embedding must be of type F32")
            if(storedEmbed.embedding !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Mismatch between query embedding and stored embeddings. Both must be of type F32")
        }
    }

    private fun handleInvalidEmbedStoreError(e: SmartScanException.InvalidEmbeddingStoreFile): SmartScanException{
        if(listOf(e.fileSize, e.expectedFileSize, e.count).all { it != null }){
            val quantRecordSize = 8 + 8 + embeddingDimension
            val f32RecordSize =  8 + 8 + embeddingDimension * 4
            val (expectedSizeOfOtherCodec, storedType) = if(quantize){
                HEADER_SIZE + e.count!! * f32RecordSize to "F32"
            }else{
                HEADER_SIZE + e.count!! * quantRecordSize to "QInt8"
            }
            if (expectedSizeOfOtherCodec == e.fileSize) {
                return SmartScanException.CodecMismatch(
                    "Codec mismatch: quantize=$quantize but the file passed stores embedding that are $storedType format. "
                )
            } 
        }
        return e
    }
}