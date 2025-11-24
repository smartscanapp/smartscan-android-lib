package com.fpf.smartscansdk.core.embeddings

interface IDetectorProvider<T> {
    fun closeSession() = Unit
    suspend fun initialize()
    fun isInitialized(): Boolean
    suspend fun detect(data: T): Pair<List<Float>, List<FloatArray>>
}