package com.fpf.smartscansdk.core.indexers

import android.content.ContentUris
import android.content.Context
import android.provider.MediaStore
import com.fpf.smartscansdk.core.embeddings.StoredEmbedding
import com.fpf.smartscansdk.core.embeddings.EmbeddingStore
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.media.getBitmapFromUri
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.core.processors.ProcessorListener
import com.fpf.smartscansdk.core.processors.MemoryOptions
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext

// ** Design Constraint**: For on-device vector search, the full index needs to be loaded in-memory (or make an Android native VectorDB)
// File-based EmbeddingStore is used over a Room version due to significant faster index loading
// Benchmarks: Memory-Mapped File loading 30-50x speed vs Room (both LiveData and Synchronous),
// These benchmarks strongly favour the  FileEmbeddingStore for optimal on-device search functionality and UX.

class ImageIndexer(
    private val embedder: ImageEmbeddingProvider,
    private val store: EmbeddingStore,
    private val maxImageSize: Int = 225,
    context: Context,
    listener: ProcessorListener<Long, Pair<Long, FloatArray>>? = null,
    memoryOptions: MemoryOptions = MemoryOptions(),
    batchSize: Int = 10,
    ): BatchProcessor<Long, Pair<Long, FloatArray>>(context, listener, memoryOptions, batchSize){


    override suspend fun onBatchComplete(context: Context, batch: List<Pair<Long, FloatArray>>) {
        val imageIdToDateMap = getImageToDateMap(context, batch.map { it.first })
        val embedsToStore = batch.map{
            val date = imageIdToDateMap[it.first]?: System.currentTimeMillis()
            StoredEmbedding(it.first, date, it.second)
        }
        store.add(embedsToStore)
        listener?.onBatchComplete(context, batch)
    }

    override suspend fun onProcess(context: Context, item: Long): Pair<Long, FloatArray> {
        val contentUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, item)
        val bitmap = getBitmapFromUri(context, contentUri, maxImageSize)
        val embedding = withContext(NonCancellable) { embedder.embed(bitmap) }
        return Pair(item, embedding)
    }

    private fun getImageToDateMap(context: Context, ids: List<Long>): Map<Long, Long> {
        val result = mutableMapOf<Long, Long>()
        val uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_ADDED
        )

        val chunkSize = 500

        ids.chunked(chunkSize).forEach { chunk ->

            val selection = "${MediaStore.Images.Media._ID} IN (${
                chunk.joinToString(",")
            })"

            context.applicationContext.contentResolver.query(
                uri,
                projection,
                selection,
                null,
                null
            )?.use { cursor ->

                val idIdx = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)

                while (cursor.moveToNext()) {
                    result[cursor.getLong(idIdx)] = cursor.getLong(dateIdx)
                }
            }
        }
        return result
    }
}