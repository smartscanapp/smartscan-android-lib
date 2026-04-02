package com.fpf.smartscansdk.core.detector

interface IDetectorProvider<T> {
    fun closeSession() = Unit
    suspend fun initialize()
    fun isInitialized(): Boolean
    suspend fun detect(data: T): Pair<List<Float>, List<FloatArray>>
}