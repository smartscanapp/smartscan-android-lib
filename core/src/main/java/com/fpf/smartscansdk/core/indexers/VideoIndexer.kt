package com.fpf.smartscansdk.core.indexers

import android.content.ContentUris
import android.content.Context
import android.provider.MediaStore
import com.fpf.smartscansdk.core.embeddings.StoredEmbedding
import com.fpf.smartscansdk.core.embeddings.EmbeddingStore
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.embedBatch
import com.fpf.smartscansdk.core.embeddings.generatePrototypeEmbedding
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.core.media.extractFramesFromVideo
import com.fpf.smartscansdk.core.processors.ProcessorListener
import com.fpf.smartscansdk.core.processors.MemoryOptions

// ** Design Constraint**: For on-device vector search, the full index needs to be loaded in-memory (or make an Android native VectorDB)
// File-based EmbeddingStore is used over a Room version due to significant faster index loading
// Benchmarks: Memory-Mapped File loading 30-50x speed vs Room (both LiveData and Synchronous),
// These benchmarks strongly favour the  FileEmbeddingStore for optimal on-device search functionality and UX.
// **IMPORTANT**: Video frame extraction can fail due to codec incompatibility

class VideoIndexer(
    private val embedder: ImageEmbeddingProvider,
    private val frameCount: Int = 10,
    private val width: Int,
    private val height: Int,
    context: Context,
    listener: ProcessorListener<Long, Pair<Long, FloatArray>>? = null,
    batchSize: Int = 10,
    memoryOptions: MemoryOptions = MemoryOptions(),
    private val store: EmbeddingStore,
    ): BatchProcessor<Long, Pair<Long, FloatArray>>(context, listener, memoryOptions, batchSize){

    override suspend fun onBatchComplete(context: Context, batch: List<Pair<Long, FloatArray>>) {
        val videoIdToDateMap = getVideoToDateMap(context, batch.map { it.first })
        val embedsToStore = batch.map{
            val date = videoIdToDateMap[it.first]?: System.currentTimeMillis()
            StoredEmbedding(it.first, date, it.second)
        }
        store.add(embedsToStore)
        listener?.onBatchComplete(context, batch)
    }

    override suspend fun onProcess(context: Context, item: Long): Pair<Long, FloatArray> {
        val contentUri = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, item)
        val frameBitmaps = extractFramesFromVideo(context, contentUri, width = width, height = height, frameCount = frameCount)?: throw IllegalStateException("Invalid frames")
        val rawEmbeddings = embedBatch(context, embedder, frameBitmaps)
        val embedding: FloatArray = generatePrototypeEmbedding(rawEmbeddings)
        return Pair(item, embedding)
    }

    private fun getVideoToDateMap(context: Context, ids: List<Long>): Map<Long, Long> {
        val result = mutableMapOf<Long, Long>()
        val uri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        val projection = arrayOf(
            MediaStore.Video.Media._ID,
            MediaStore.Video.Media.DATE_ADDED
        )

        val chunkSize = 500

        ids.chunked(chunkSize).forEach { chunk ->

            val selection = "${MediaStore.Video.Media._ID} IN (${
                chunk.joinToString(",")
            })"

            context.applicationContext.contentResolver.query(
                uri,
                projection,
                selection,
                null,
                null
            )?.use { cursor ->

                val idIdx = cursor.getColumnIndexOrThrow(MediaStore.Video.Media._ID)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATE_ADDED)

                while (cursor.moveToNext()) {
                    result[cursor.getLong(idIdx)] = cursor.getLong(dateIdx)
                }
            }
        }

        return result
    }

}