package com.fpf.smartscansdk.core.embeddings

import android.graphics.Bitmap


interface IEmbeddingProvider<T> {
    val embeddingDim: Int
    suspend fun initialize()
    fun isInitialized(): Boolean
    fun closeSession() = Unit
    suspend fun embed(data: T): FloatArray
}

interface TextEmbeddingProvider : IEmbeddingProvider<String> {
    val maxTokens: Int
}
typealias ImageEmbeddingProvider = IEmbeddingProvider<Bitmap>

