package com.fpf.smartscansdk.core.indexers

import android.content.ContentUris
import android.content.Context
import android.provider.MediaStore
import com.fpf.smartscansdk.core.embeddings.StoredEmbedding
import com.fpf.smartscansdk.core.embeddings.IEmbeddingStore
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.generatePrototypeEmbedding
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.core.media.extractFramesFromVideo
import com.fpf.smartscansdk.core.processors.IProcessorListener
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
    listener: IProcessorListener<Long, StoredEmbedding>? = null,
    batchSize: Int = 10,
    memoryOptions: MemoryOptions = MemoryOptions(),
    private val store: IEmbeddingStore,
    ): BatchProcessor<Long, StoredEmbedding>(context, listener, memoryOptions, batchSize){

    companion object {
        const val INDEX_FILENAME = "video_index.bin"
    }

    override suspend fun onBatchComplete(context: Context, batch: List<StoredEmbedding>) {
        store.add(batch)
        listener?.onBatchComplete(context, batch)
    }

    override suspend fun onProcess(context: Context, item: Long): StoredEmbedding {
        val contentUri = ContentUris.withAppendedId(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI, item
        )
        val frameBitmaps = extractFramesFromVideo(context, contentUri, width = width, height = height, frameCount = frameCount)

        if(frameBitmaps == null) throw IllegalStateException("Invalid frames")

        val rawEmbeddings = embedder.embedBatch(frameBitmaps)
        val embedding: FloatArray = generatePrototypeEmbedding(rawEmbeddings)

        return StoredEmbedding(
            id = item,
            date = System.currentTimeMillis(),
            embedding = embedding
        )
    }

}