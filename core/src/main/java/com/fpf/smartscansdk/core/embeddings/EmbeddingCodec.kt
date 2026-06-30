package com.fpf.smartscansdk.core.embeddings

import java.io.File
import java.nio.channels.FileChannel

internal interface EmbeddingCodec {
    suspend fun writeReplace(embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>, outputFile: File)
    suspend fun read(file: File): Pair<LinkedHashMap<Long, StoredEmbedding>, MutableMap<Long, Long>>
    suspend  fun append(file: File, embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>): Int
    suspend fun update(file: File, embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>): Int
    suspend fun readHeader(channel: FileChannel): Int
    suspend fun writeHeader(channel: FileChannel, embeddingCount: Int)
}