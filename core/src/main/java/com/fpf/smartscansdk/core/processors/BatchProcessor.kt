package com.fpf.smartscansdk.core.processors

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicInteger

abstract class BatchProcessor<Input, Output>(
    private val context: Context,
    protected val listener: ProcessorListener<Input>? = null,
    private val memoryOptions: MemoryOptions = MemoryOptions(),
    val batchSize: Int = 10
) {
    companion object {
        const val TAG = "BatchProcessor"
    }

    suspend fun run(items: List<Input>): ProcessorResult = withContext(Dispatchers.IO) {
        val processedCount = AtomicInteger(0)
        val startTime = System.currentTimeMillis()
        var totalSuccess = 0

        try {
            if (items.isEmpty()) {
                Log.w(TAG, "No items to process.")
                val processorResult = ProcessorResult.Success()
                listener?.onComplete(context.applicationContext, processorResult)
                return@withContext processorResult
            }

            val memoryUtils = Memory(context.applicationContext, memoryOptions)

            listener?.onActive(context.applicationContext)

            for (batch in items.chunked(batchSize)) {
                val currentConcurrency = memoryUtils.calculateConcurrencyLevel()
                val semaphore = Semaphore(currentConcurrency)

                val deferredResults = batch.map { item ->
                    async {
                        semaphore.withPermit {
                            try {
                                val output = onProcess(context.applicationContext, item)
                                item to Result.success(output)
                            } catch (e: Exception) {
                                item to Result.failure(e)
                            }finally {
                                val current = processedCount.incrementAndGet()
                                val progress = current.toFloat() / items.size
                                listener?.onProgress(context.applicationContext, progress)
                            }
                        }
                    }
                }

                val successfulResults = mutableListOf<Output>()

                try {
                    for (deferred in deferredResults) {
                        val result = deferred.await()
                        if (result.second.isSuccess) {
                            successfulResults += result.second.getOrThrow()
                        } else {
                            val error = result.second.exceptionOrNull() as Exception
                            listener?.onError(context.applicationContext, error, result.first)
                        }
                    }
                } catch (e: Exception) {
                    // Fatal item error, cancel all remaining in batch
                    deferredResults.forEach {
                        it.cancel()
                    }
                    throw e
                }

                totalSuccess += successfulResults.size
                onBatchComplete(context.applicationContext, successfulResults)
            }

            val endTime = System.currentTimeMillis()
            val processorResult = ProcessorResult.Success(totalSuccess, timeElapsed = endTime - startTime)

            listener?.onComplete(context.applicationContext, processorResult)
            processorResult
        }
        catch (e: CancellationException) {
            throw e
        }
        catch (e: Exception) {
            val processorResult = ProcessorResult.Failure(
                totalProcessed = totalSuccess,
                timeElapsed = System.currentTimeMillis() - startTime,
                error = e
            )
            listener?.onFail(context.applicationContext, processorResult)
            processorResult
        }
    }

    // Subclasses must implement this
    protected abstract suspend fun onProcess(context: Context, item: Input): Output

    // Forces all SDK users to consciously handle batch events rather than optionally relying on listeners.
    // This can prevent subtle bugs where batch-level behavior is forgotten.
    // Subclasses can optionally delegate to listener (client app) by simply calling listener.onBatchComplete in implementation
    protected abstract suspend fun onBatchComplete(context: Context, batch: List<Output>)

}

