package com.fpf.smartscansdk.core.processors

import android.content.Context

interface IProcessorListener<Input, Output> {
    suspend fun onActive(context: Context) = Unit
    suspend fun onBatchComplete(context: Context, batch: List<Output>) = Unit
    suspend fun onComplete(context: Context, metrics: Metrics.Success) = Unit
    suspend fun onProgress(context: Context, progress: Float) = Unit
    fun onError(context: Context, error: Exception, item: Input) = Unit
    suspend fun onFail(context: Context, failureMetrics: Metrics.Failure) = Unit
}