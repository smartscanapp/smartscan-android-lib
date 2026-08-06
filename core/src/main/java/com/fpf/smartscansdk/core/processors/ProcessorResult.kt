package com.fpf.smartscansdk.core.processors

sealed interface ProcessorResult {
    val totalProcessed: Int
    val timeElapsed: Long
    data class Success(override val totalProcessed: Int = 0, override val timeElapsed: Long = 0L) : ProcessorResult
    data class Failure(override val totalProcessed: Int, override val timeElapsed: Long, val error: Exception) : ProcessorResult
}