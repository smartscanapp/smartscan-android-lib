package com.fpf.smartscansdk.core.processors

import android.content.Context

interface ProcessorListener<Input> {
    suspend fun onActive(context: Context) = Unit
    suspend fun onComplete(context: Context, result: ProcessorResult.Success) = Unit
    suspend fun onProgress(context: Context, progress: Float) = Unit
    suspend fun onError(context: Context, error: Exception, item: Input) = Unit
    suspend fun onFail(context: Context, result: ProcessorResult.Failure) = Unit

    suspend fun onCancel(context: Context) = Unit

}