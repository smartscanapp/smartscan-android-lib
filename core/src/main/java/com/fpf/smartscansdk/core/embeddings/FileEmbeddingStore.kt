package com.fpf.smartscansdk.core.embeddings

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
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

    private var cache: LinkedHashMap<Long, Embedding> = LinkedHashMap()
    private var cachedIds: List<Long>? = null

    override val exists: Boolean get() = file.exists()

    // prevent OOM in FileEmbeddingStore.save() by batching writes
    private suspend fun save(embeddingsList: List<Embedding>): Unit = withContext(Dispatchers.IO) {
        if (embeddingsList.isEmpty()) return@withContext

        cache = LinkedHashMap(embeddingsList.associateBy { it.id })

        FileOutputStream(file).channel.use { channel ->
            val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(embeddingsList.size)
            header.flip()
            channel.write(header)

            val batchSize = 1000 // number of embeddings per batch
            var index = 0

            while (index < embeddingsList.size) {
                val end = minOf(index + batchSize, embeddingsList.size)
                val batch = embeddingsList.subList(index, end)

                // Allocate a smaller buffer for this batch
                val batchBuffer = ByteBuffer.allocate(batch.size * (8 + 8 + embeddingDimension * 4))
                    .order(ByteOrder.LITTLE_ENDIAN)

                for (embedding in batch) {
                    if (embedding.embeddings.size != embeddingDimension) {
                        throw IllegalArgumentException("Embedding dimension must be $embeddingDimension")
                    }
                    batchBuffer.putLong(embedding.id)
                    batchBuffer.putLong(embedding.date)
                    for (f in embedding.embeddings) {
                        batchBuffer.putFloat(f)
                    }
                }

                batchBuffer.flip()
                channel.write(batchBuffer)
                index = end
            }
        }
    }

    // This explicitly makes clear the design constraints that requires the full index to be loaded in memory
    override suspend fun get(): List<Embedding> = withContext(Dispatchers.IO){
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
                cache[id] = Embedding(id, date, floats)
            }
            cache.values.toList()
        }
    }

    suspend fun get(ids: List<Long>): List<Embedding> = withContext(Dispatchers.IO) {
        val embeddings = mutableListOf<Embedding>()

        for (id in ids) {
            cache.get(id)?.let { embeddings.add(it) }
        }
        embeddings
    }

    override suspend fun add(newEmbeddings: List<Embedding>): Unit = withContext(Dispatchers.IO) {
        val filteredNewEmbeddings = newEmbeddings.filterNot { it.id in cache }
        if (filteredNewEmbeddings.isEmpty()) return@withContext

        if (!file.exists()) {
            save(filteredNewEmbeddings)
            return@withContext
        }

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Read the 4-byte header as little-endian
            val headerBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            channel.position(0)
            val read = channel.read(headerBuf)
            if (read != 4) {
                throw IOException("Failed to read header count (file too small/corrupted)")
            }
            headerBuf.flip()
            val existingCount = headerBuf.int

            // Basic validation: each existing entry is at least id(8)+date(8)+EMBEDDING_LEN*4
            val minEntryBytes = 8 + 8 + embeddingDimension * 4
            val maxCountFromSize = (channel.size() / minEntryBytes).toInt()
            if (existingCount < 0 || existingCount > maxCountFromSize + 10_000) {
                throw IOException("Corrupt embeddings header: count=$existingCount, fileSize=${channel.size()}")
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
                if (embedding.embeddings.size != embeddingDimension) {
                    throw IllegalArgumentException("Embedding dimension must be $embeddingDimension")
                }
                val entryBytes = (8 + 8) + embeddingDimension * 4
                val buf = ByteBuffer.allocate(entryBytes).order(ByteOrder.LITTLE_ENDIAN)
                buf.putLong(embedding.id)
                buf.putLong(embedding.date)
                for (f in embedding.embeddings) buf.putFloat(f)
                buf.flip()
                while (buf.hasRemaining()) {
                    channel.write(buf)
                }
                cache[embedding.id] = embedding
            }
            channel.force(false)
        }
    }

    override suspend fun remove(ids: List<Long>): Unit = withContext(Dispatchers.IO) {
        if (ids.isEmpty()) return@withContext

        try {
            var removedCount = 0
            for (id in ids) {
                if (cache.remove(id) != null) removedCount++
            }

            if (removedCount > 0) {
                save(cache.values.toList())
                Log.i(TAG, "Removed $removedCount stale embeddings")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error Removing embeddings", e)
        }
    }


    override fun clear(){
        cache.clear()
    }


    override suspend fun query(embedding: FloatArray, topK: Int, threshold: Float): List<Long> {
        cachedIds = null // clear on new search

        val storedEmbeddings = get()

        if (storedEmbeddings.isEmpty()) return emptyList()

        val similarities = getSimilarities(embedding, storedEmbeddings.map { it.embeddings })
        val resultIndices = getTopN(similarities, topK, threshold)

        if (resultIndices.isEmpty()) return emptyList()

        val results = resultIndices.map{idx -> storedEmbeddings[idx].id }
        cachedIds = results
        return results
    }

    suspend fun query(start: Int, end: Int): List<Long> {
        val ids = cachedIds ?: return emptyList()
        val s = start.coerceAtLeast(0)
        val e = end.coerceAtMost(ids.size)
        if (s >= e) return emptyList()

        val batch = get(ids.subList(s, e))
        return batch.map { it.id }
    }

}