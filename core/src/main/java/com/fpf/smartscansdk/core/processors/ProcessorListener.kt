package com.fpf.smartscansdk.core.processors

import android.content.Context

interface ProcessorListener<Input> {
    suspend fun onActive() = Unit
    suspend fun onComplete(result: ProcessorResult.Success) = Unit
    suspend fun onProgress( progress: Float) = Unit
    suspend fun onError(error: Exception, item: Input) = Unit
    suspend fun onFail(result: ProcessorResult.Failure) = Unit

    suspend fun onCancel() = Unit

}