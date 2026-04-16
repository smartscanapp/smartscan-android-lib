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
    IEmbeddingStore {

    companion object {
        const val TAG = "FileEmbeddingStore"
    }

    private val fileMutex = Mutex()

    private var cache: LinkedHashMap<Long, StoredEmbedding> = LinkedHashMap() // initialised in get and only updated in save
    private var idToFileOffsetIndex: MutableMap<Long, Long> = mutableMapOf() // id -> file offset

    override val exists: Boolean get() = file.exists()

    private val recordSize = (8 + 8) + embeddingDimension * 4
    private val headerSize = 4



    private suspend fun save(embeddingsList: List<StoredEmbedding>): Unit = withContext(Dispatchers.IO) {
        if (embeddingsList.isEmpty()) return@withContext

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
            if (idToFileOffsetIndex.isEmpty()) load() // initialise index and cache

            val filteredNewEmbeddings = embeddings.filterNot { it.id in idToFileOffsetIndex }
            if (filteredNewEmbeddings.isEmpty()) return@withContext 0

            if (!file.exists()) {
                save(filteredNewEmbeddings)
                return@withContext filteredNewEmbeddings.size
            }

            RandomAccessFile(file, "rw").use { raf ->
                val channel = raf.channel
                val existingCount = readAndValidateHeader(channel)
                val newCount = existingCount + filteredNewEmbeddings.size

                // Write the updated count back as little-endian
                val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
                headerBuf.putInt(newCount).flip()
                channel.position(0)
                while (headerBuf.hasRemaining()) channel.write(headerBuf)

                // Move to the end to append new entries
                var nextOffset = channel.size()
                channel.position(nextOffset)

                for (embedding in filteredNewEmbeddings) {
                    if (embedding.embedding.size != embeddingDimension) {
                        throw SmartScanException.InvalidEmbeddingDimension(
                            "Embedding dimension mismatch. Expected $embeddingDimension, got ${embedding.embedding.size}"
                        )
                    }

                    val buf = ByteBuffer.allocate(recordSize).order(ByteOrder.LITTLE_ENDIAN)
                    buf.putLong(embedding.id)
                    buf.putLong(embedding.date)
                    for (f in embedding.embedding) buf.putFloat(f)
                    buf.flip()

                    while (buf.hasRemaining()) {
                        channel.write(buf)
                    }

                    // update in-memory file offset index for the newly appended entry and cache
                    idToFileOffsetIndex[embedding.id] = nextOffset

                    // Only add items to cache if it's not empty e.g after get() call, to keep it synchronized
                    // This prevents unnecessarily keeping embeddings in memory
                    if(cache.isNotEmpty()){
                        cache[embedding.id] = embedding
                    }
                    nextOffset += recordSize
                }

                channel.force(false)
            }

            filteredNewEmbeddings.size
        }
    }

    override suspend fun remove(ids: List<Long>): Int = fileMutex.withLock {
        withContext(Dispatchers.IO) {
            if (ids.isEmpty()) return@withContext 0

            if (idToFileOffsetIndex.isEmpty()) {
                load()
            }

            val idsToRemove = ids.toSet()
            val existingIdsToRemove = idsToRemove.filterTo(mutableSetOf()) { idToFileOffsetIndex.containsKey(it) }
            if (existingIdsToRemove.isEmpty()) return@withContext 0

            val rebuiltIndex = LinkedHashMap<Long, Long>()

            RandomAccessFile(file, "rw").use { raf ->
                val channel = raf.channel
                val existingCount = readAndValidateHeader(channel)
                val recordBuffer = ByteBuffer.allocate(recordSize).order(ByteOrder.LITTLE_ENDIAN)
                var readPos = headerSize.toLong()
                var writePos = headerSize.toLong()
                var keptCount = 0

                repeat(existingCount) {
                    recordBuffer.clear()
                    channel.position(readPos)

                    while (recordBuffer.hasRemaining()) {
                        val n = channel.read(recordBuffer)
                        if (n < 0) {
                            throw SmartScanException.CorruptedEmbeddingStoreFile("Unexpected EOF while removing embeddings")
                        }
                    }

                    recordBuffer.flip()
                    val id = recordBuffer.getLong(0)

                    if (id !in idsToRemove) {
                        channel.position(writePos)
                        while (recordBuffer.hasRemaining()) {
                            channel.write(recordBuffer)
                        }
                        rebuiltIndex[id] = writePos
                        writePos += recordSize.toLong()
                        keptCount++
                    }

                    readPos += recordSize.toLong()
                }

                val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
                headerBuf.putInt(keptCount).flip()
                channel.position(0)
                while (headerBuf.hasRemaining()) {
                    channel.write(headerBuf)
                }

                channel.truncate(writePos)
                channel.force(false)
            }

            idToFileOffsetIndex = rebuiltIndex

            for (id in existingIdsToRemove) {
                cache.remove(id)
            }

            existingIdsToRemove.size
        }
    }

    override fun clear(){
        cache.clear()
        idToFileOffsetIndex.clear()
    }

    override suspend fun query(embedding: FloatArray, topK: Int, threshold: Float, ids: Set<Long>): List<Long> {
        val storedEmbeddings = if (ids.isNotEmpty()) get().filter { it.id in ids } else get()

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

                    // Only update in cache if it's not empty e.g after get() call, to keep it synchronized
                    // This prevents unnecessarily keeping embeddings in memory
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