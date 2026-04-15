package com.fpf.smartscansdk.core.embeddings

import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
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
    ):
    IEmbeddingStore {

    companion object {
        const val TAG = "FileEmbeddingStore"
    }

    private var cache: LinkedHashMap<Long, StoredEmbedding> = LinkedHashMap()

    override val exists: Boolean get() = file.exists()

    private suspend fun save(embeddingsList: List<StoredEmbedding>): Unit = withContext(Dispatchers.IO) {
        if (embeddingsList.isEmpty()) return@withContext

        cache = LinkedHashMap(embeddingsList.associateBy { it.id })

        FileOutputStream(file).channel.use { channel ->
            val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(embeddingsList.size)
            header.flip()
            channel.write(header)

            val batchSize = 1000
            var index = 0

            while (index < embeddingsList.size) {
                val end = minOf(index + batchSize, embeddingsList.size)
                val batch = embeddingsList.subList(index, end)

                // Allocate a smaller buffer for this batch
                val batchBuffer = ByteBuffer.allocate(batch.size * (8 + 8 + embeddingDimension * 4))
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
                }

                batchBuffer.flip()
                channel.write(batchBuffer)
                index = end
            }
        }
    }

    override suspend fun get(): List<StoredEmbedding> = withContext(Dispatchers.IO){
        if (cache.isNotEmpty()) return@withContext cache.values.toList()

        FileInputStream(file).channel.use { ch ->
            val fileSize = ch.size()
            val buffer = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize).order(ByteOrder.LITTLE_ENDIAN)

            val count = buffer.int

            repeat(count) {
                val id = buffer.long
                val date = buffer.long
                val floats = FloatArray(embeddingDimension)
                val fb = buffer.asFloatBuffer()
                fb.get(floats)
                buffer.position(buffer.position() + embeddingDimension * 4)
                cache[id] = StoredEmbedding(id, date, floats)
            }
            cache.values.toList()
        }
    }

    suspend fun get(ids: List<Long>): List<StoredEmbedding> = withContext(Dispatchers.IO) {
        if(cache.isEmpty()) cache = LinkedHashMap(get().associateBy { it.id })
        val storedEmbeddings = mutableListOf<StoredEmbedding>()

        for (id in ids) {
            cache.get(id)?.let { storedEmbeddings.add(it) }
        }
        storedEmbeddings
    }

    override suspend fun add(embeddings: List<StoredEmbedding>): Int = withContext(Dispatchers.IO) {
        val filteredNewEmbeddings = embeddings.filterNot { it.id in cache }
        if (filteredNewEmbeddings.isEmpty()) return@withContext 0

        if (!file.exists()) {
            save(filteredNewEmbeddings)
            return@withContext filteredNewEmbeddings.size
        }

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Read the 4-byte header as little-endian
            val headerBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            channel.position(0)
            val read = channel.read(headerBuf)
            if (read != 4) {
                throw SmartScanException.CorruptedEmbeddingStoreFile("Failed to read header count")
            }
            headerBuf.flip()
            val existingCount = headerBuf.int

            // Basic validation: each existing entry is at least id(8)+date(8)+EMBEDDING_LEN*4
            val minEntryBytes = 8 + 8 + embeddingDimension * 4
            val maxCountFromSize = (channel.size() / minEntryBytes).toInt()
            if (existingCount < 0 || existingCount > maxCountFromSize + 10_000) {
                throw SmartScanException.CorruptedEmbeddingStoreFile("Corrupt embeddings header: count=$existingCount, fileSize=${channel.size()}")
            }

            val newCount = existingCount + filteredNewEmbeddings.size

            // Write the updated count back as little-endian
            headerBuf.clear()
            headerBuf.putInt(newCount).flip()
            channel.position(0)
            while (headerBuf.hasRemaining()) channel.write(headerBuf)

            // Move to the end to append new entries
            channel.position(channel.size())

            for (embedding in filteredNewEmbeddings) {
                if (embedding.embedding.size != embeddingDimension) {
                    throw SmartScanException.InvalidEmbeddingDimension("Embedding dimension mismatch. Expected $embeddingDimension, got ${embedding.embedding.size}")
                }
                val entryBytes = (8 + 8) + embeddingDimension * 4
                val buf = ByteBuffer.allocate(entryBytes).order(ByteOrder.LITTLE_ENDIAN)
                buf.putLong(embedding.id)
                buf.putLong(embedding.date)
                for (f in embedding.embedding) buf.putFloat(f)
                buf.flip()
                while (buf.hasRemaining()) {
                    channel.write(buf)
                }
                cache[embedding.id] = embedding
            }
            channel.force(false)
        }
        filteredNewEmbeddings.size
    }

    override suspend fun remove(ids: List<Long>): Int = withContext(Dispatchers.IO) {
        var removedCount = 0
        if (ids.isEmpty()) return@withContext 0
        for (id in ids) {
            if (cache.remove(id) != null) removedCount++
        }

        if (removedCount > 0) {
            save(cache.values.toList())
        }
        removedCount
    }

    override fun clear(){
        cache.clear()
    }


    override suspend fun query(embedding: FloatArray, topK: Int, threshold: Float, ids: Set<Long>): List<Long> {
        val storedEmbeddings = if (ids.isNotEmpty()) get().filter { it.id in ids } else get()

        if (storedEmbeddings.isEmpty()) return emptyList()

        val similarities = getSimilarities(embedding, storedEmbeddings.map { it.embedding })
        val resultIndices = getTopN(similarities, topK, threshold)

        if (resultIndices.isEmpty()) return emptyList()
        return resultIndices.map{idx -> storedEmbeddings[idx].id }
    }

    override suspend fun update(embeddings: List<StoredEmbedding>): Int = withContext(Dispatchers.IO) {
        var updatedCount = 0
        if (embeddings.isEmpty()) return@withContext updatedCount

        for (emb in embeddings) {
            if (emb.embedding.size != embeddingDimension) {
                throw SmartScanException.InvalidEmbeddingDimension("Embedding dimension mismatch. Expected $embeddingDimension, got ${emb.embedding.size}")
            }

            val existing = cache[emb.id]
            if (existing != null) {
                cache[emb.id] = emb
                ++updatedCount
            }
        }

        if (updatedCount > 0) {
            save(cache.values.toList())
        }

        updatedCount
    }
}