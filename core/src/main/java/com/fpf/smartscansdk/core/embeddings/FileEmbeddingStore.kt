package com.fpf.smartscansdk.core.embeddings

import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.collections.map

class FileEmbeddingStore(
    private val file: File,
    private val embeddingDimension: Int,
) :
    EmbeddingStore {

    companion object {
        const val TAG = "FileEmbeddingStore"
    }

    private val fileMutex = Mutex()

    private var cache: LinkedHashMap<Long, StoredEmbedding> = LinkedHashMap() // initialised in get and only updated in save
    private var idToFileOffsetIndex: MutableMap<Long, Long> = mutableMapOf() // id -> file offset

    override val exists: Boolean get() = file.exists()
    private val recordSize = (8 + 8) + embeddingDimension * 4
    private val headerSize = 4


    override suspend fun save(): Unit = withContext(Dispatchers.IO) {
        val embeddingsList = get()
        if(embeddingsList.isEmpty()) return@withContext

        FileOutputStream(file).channel.use { channel ->
            val header = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(embeddingsList.size)
            header.flip()
            channel.write(header)

            val batchSize = 1000
            var index = 0
            var offset = headerSize

            while (index < embeddingsList.size) {
                val end = minOf(index + batchSize, embeddingsList.size)
                val batch = embeddingsList.subList(index, end)

                // Allocate a smaller buffer for this batch
                val batchBuffer = ByteBuffer.allocate(batch.size * recordSize)
                    .order(ByteOrder.LITTLE_ENDIAN)

                for (embedding in batch) {
                    if (embedding.embedding.size != embeddingDimension) {
                        throw SmartScanException.InvalidEmbeddingDimension("Embedding dimension mismatch. Expected $embeddingDimension, got ${embedding.embedding.size}")
                    }
                    batchBuffer.putLong(embedding.id)
                    batchBuffer.putLong(embedding.date)
                    for (f in embedding.embedding) {
                        batchBuffer.putFloat(f)
                    }
                    idToFileOffsetIndex[embedding.id] = offset.toLong()
                    offset += recordSize
                }
                batchBuffer.flip()
                channel.write(batchBuffer)
                index = end
            }
        }
    }

    private suspend fun load(): LinkedHashMap<Long, StoredEmbedding> = withContext(Dispatchers.IO) {
        val map = LinkedHashMap<Long, StoredEmbedding>()
        val idx = mutableMapOf<Long, Long>()

        if (!file.exists()) return@withContext map

        FileInputStream(file).channel.use { ch ->
            val size = ch.size()
            if (size < headerSize) {
                throw SmartScanException.CorruptedEmbeddingStoreFile("File too small to contain header")
            }

            val buffer = ch.map(FileChannel.MapMode.READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN)

            val count = buffer.int
            val expectedSize = headerSize.toLong() + count.toLong() * recordSize.toLong()

            if (count < 0 || expectedSize > size) {
                throw SmartScanException.CorruptedEmbeddingStoreFile(
                    "Corrupt embeddings header: count=$count, fileSize=$size"
                )
            }

            var offset = headerSize.toLong()

            repeat(count) {
                val id = buffer.long
                val date = buffer.long
                val floats = FloatArray(embeddingDimension)
                val fb = buffer.asFloatBuffer()
                fb.get(floats)
                buffer.position(buffer.position() + embeddingDimension * 4)

                map[id] = StoredEmbedding(id, date, floats)
                idx[id] = offset
                offset += recordSize.toLong()
            }
        }

        idToFileOffsetIndex = idx
        map
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

            if (idToFileOffsetIndex.isEmpty() && file.exists()) {
                load()
            }

            val filteredNewEmbeddings = embeddings.filterNot { it.id in idToFileOffsetIndex }
            if (filteredNewEmbeddings.isEmpty()) return@withContext 0

            for (embedding in filteredNewEmbeddings) {
                if (embedding.embedding.size != embeddingDimension) {
                    throw SmartScanException.InvalidEmbeddingDimension(
                        "Embedding dimension mismatch. Expected $embeddingDimension, got ${embedding.embedding.size}"
                    )
                }
            }

            RandomAccessFile(file, "rw").use { raf ->
                val channel = raf.channel

                val fileExistsAndHasContent = channel.size() >= headerSize.toLong()

                val existingCount = if (fileExistsAndHasContent) {
                    readAndValidateHeader(channel)
                } else {
                    0
                }

                val newCount = existingCount + filteredNewEmbeddings.size

                // Move to the end (append mode) or start of data section if new file
                val nextOffset = if (fileExistsAndHasContent) {
                    channel.size()
                } else {
                    headerSize.toLong()
                }

                channel.position(nextOffset)

                val targetChunkBytes = 4 * 1024 * 1024
                val chunkCapacity = maxOf(
                    recordSize,
                    (targetChunkBytes / recordSize).coerceAtLeast(1) * recordSize
                )
                val writeBuffer = ByteBuffer.allocateDirect(chunkCapacity).order(ByteOrder.LITTLE_ENDIAN)

                fun flushBuffer() {
                    writeBuffer.flip()
                    while (writeBuffer.hasRemaining()) {
                        channel.write(writeBuffer)
                    }
                    writeBuffer.clear()
                }

                for (embedding in filteredNewEmbeddings) {
                    if (writeBuffer.remaining() < recordSize) {
                        flushBuffer()
                    }

                    writeBuffer.putLong(embedding.id)
                    writeBuffer.putLong(embedding.date)
                    for (f in embedding.embedding) writeBuffer.putFloat(f)
                }

                if (writeBuffer.position() > 0) {
                    flushBuffer()
                }

                // Write updated count back as little-endian
                val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
                headerBuf.putInt(newCount)
                headerBuf.flip()
                channel.position(0)
                while (headerBuf.hasRemaining()) {
                    channel.write(headerBuf)
                }

                channel.force(false)

                // update in-memory file offset index for the newly appended entry and cache
                filteredNewEmbeddings.forEachIndexed { index, embedding ->
                    idToFileOffsetIndex[embedding.id] = nextOffset + (index.toLong() * recordSize)
                }

                // Only add items to cache if it's not empty e.g after get() call, to keep it synchronized.
                // This prevents edge cases that could result in partial cache overwriting on-disk data
                // It also prevents unnecessarily keeping embeddings in memory
                if (cache.isNotEmpty()) {
                    for (embedding in filteredNewEmbeddings) {
                        cache[embedding.id] = embedding
                    }
                }
            }

            filteredNewEmbeddings.size
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

    override fun clear(){
        cache.clear()
        idToFileOffsetIndex.clear()
    }

    override suspend fun query(embedding: FloatArray, topK: Int, threshold: Float, ids: Set<Long>, startDate: Long?, endDate: Long?): List<Long> {
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

        if (storedEmbeddings.isEmpty()) return emptyList()

        val similarities = getSimilarities(embedding, storedEmbeddings.map { it.embedding })
        val resultIndices = getTopN(similarities, topK, threshold)

        if (resultIndices.isEmpty()) return emptyList()
        return resultIndices.map{idx -> storedEmbeddings[idx].id }
    }

    override suspend fun update(embeddings: List<StoredEmbedding>): Int = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            var updatedCount = 0
            if (embeddings.isEmpty()) return@withContext updatedCount

            if (idToFileOffsetIndex.isEmpty()) {
                load()
            }

            RandomAccessFile(file, "rw").use { raf ->
                val channel = raf.channel

                for (emb in embeddings) {
                    if (emb.embedding.size != embeddingDimension) {
                        throw SmartScanException.InvalidEmbeddingDimension(
                            "Embedding dimension mismatch. Expected $embeddingDimension, got ${emb.embedding.size}"
                        )
                    }
                    val offset = idToFileOffsetIndex[emb.id] ?: continue
                    val buf = ByteBuffer.allocate(recordSize).order(ByteOrder.LITTLE_ENDIAN)
                    buf.putLong(emb.id)
                    buf.putLong(emb.date)
                    for (f in emb.embedding) buf.putFloat(f)
                    buf.flip()

                    channel.position(offset)
                    while (buf.hasRemaining()) {
                        channel.write(buf)
                    }
                    // Only add items to cache if it's not empty e.g after get() call, to keep it synchronized.
                    // This prevents edge cases that could result in partial cache overwriting on-disk data
                    // It also prevents unnecessarily keeping embeddings in memory
                    if(cache.isNotEmpty()){
                        cache[emb.id] = emb
                    }
                    updatedCount++
                }

                channel.force(false)
            }
            updatedCount
        }
    }

    private fun readAndValidateHeader(channel: FileChannel): Int{
        val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
        channel.position(0)
        val read = channel.read(headerBuf)
        if (read != headerSize) {
            throw SmartScanException.CorruptedEmbeddingStoreFile("Failed to read header count")
        }
        headerBuf.flip()

        val size = channel.size()
        val existingCount = headerBuf.int
        val maxCountFromSize = (size / recordSize)
        if (existingCount !in 0..maxCountFromSize) {
            throw SmartScanException.CorruptedEmbeddingStoreFile("Corrupt embeddings header: count=$existingCount, fileSize=${size}")
        }
        return existingCount
    }
}