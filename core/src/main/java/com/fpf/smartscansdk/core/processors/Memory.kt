package com.fpf.smartscansdk.core.processors

import android.app.ActivityManager
import android.content.Context

data class MemoryOptions(
    val lowMemoryThreshold: Long = 800L * 1024 * 1024,
    val highMemoryThreshold: Long = 1_600L * 1024 * 1024,
    val minConcurrency: Int = 1,
    val maxConcurrency: Int = 4
)
class Memory(
    private val context: Context,
    memoryOptions: MemoryOptions? = null
) {
    private val memoryOptions = MemoryOptions().copy(
        lowMemoryThreshold = memoryOptions?.lowMemoryThreshold ?: MemoryOptions().lowMemoryThreshold,
        highMemoryThreshold = memoryOptions?.highMemoryThreshold ?: MemoryOptions().highMemoryThreshold,
        minConcurrency = memoryOptions?.minConcurrency ?: MemoryOptions().minConcurrency,
        maxConcurrency = memoryOptions?.maxConcurrency ?: MemoryOptions().maxConcurrency,
    )

    fun getFreeMemory(): Long {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        return memoryInfo.availMem
    }

    fun calculateConcurrencyLevel(): Int {
        val freeMemory = getFreeMemory()
        return when {
            freeMemory < memoryOptions.lowMemoryThreshold -> memoryOptions.minConcurrency
            freeMemory >= memoryOptions.highMemoryThreshold -> memoryOptions.maxConcurrency
            else -> {
                val proportion = (freeMemory - memoryOptions.lowMemoryThreshold).toDouble() /
                        (memoryOptions.highMemoryThreshold - memoryOptions.lowMemoryThreshold)
                (memoryOptions.minConcurrency + proportion * (memoryOptions.maxConcurrency - memoryOptions.minConcurrency)).toInt().coerceAtLeast(memoryOptions.minConcurrency)
            }
        }
    }
}
