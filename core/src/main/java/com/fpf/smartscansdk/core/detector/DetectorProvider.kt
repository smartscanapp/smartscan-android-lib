package com.fpf.smartscansdk.core.detector

import android.graphics.Bitmap

interface DetectorProvider {
    fun closeSession() = Unit
    suspend fun initialize()
    fun isInitialized(): Boolean
    suspend fun detect(input: Bitmap): Pair<List<Float>, List<FloatArray>>
}