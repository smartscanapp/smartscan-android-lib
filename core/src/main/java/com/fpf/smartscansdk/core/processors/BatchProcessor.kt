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

// For BatchProcessor’s use case—long-running, batched,  asynchronous processing—the Application context should be used.
abstract class BatchProcessor<Input, Output>(
    private val context: Context,
    protected val listener: IProcessorListener<Input, Output>? = null,
    private val memoryOptions: MemoryOptions = MemoryOptions(),
    val batchSize: Int = 10
) {
    companion object {
        const val TAG = "BatchProcessor"
    }

    open suspend fun run(items: List<Input>): Metrics = withContext(Dispatchers.IO) {
        val processedCount = AtomicInteger(0)
        val startTime = System.currentTimeMillis()
        var totalSuccess = 0

        try {
            if (items.isEmpty()) {
                Log.w(TAG, "No items to process.")
                val metrics = Metrics.Success()
                listener?.onComplete(context.applicationContext, metrics)
                return@withContext metrics
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
                                output
                            } catch (e: Exception) {
                                listener?.onError(context.applicationContext, e, item)
                                null
                            }finally {
                                val current = processedCount.incrementAndGet()
                                val progress = current.toFloat() / items.size
                                listener?.onProgress(context.applicationContext, progress)
                            }
                        }
                    }
                }

                val outputBatch = deferredResults.mapNotNull { it.await() }
                totalSuccess += outputBatch.size
                onBatchComplete(context.applicationContext, outputBatch)
            }

            val endTime = System.currentTimeMillis()
            val metrics = Metrics.Success(totalSuccess, timeElapsed = endTime - startTime)

            listener?.onComplete(context.applicationContext, metrics)
            metrics
        }
        catch (e: CancellationException) {
            throw e
        }
        catch (e: Exception) {
            val metrics = Metrics.Failure(
                processedBeforeFailure = totalSuccess,
                timeElapsed = System.currentTimeMillis() - startTime,
                error = e
            )
            listener?.onFail(context.applicationContext, metrics)
            metrics
        }
    }

    // Subclasses must implement this
    protected abstract suspend fun onProcess(context: Context, item: Input): Output

    // Forces all SDK users to consciously handle batch events rather than optionally relying on listeners.
    // This can prevent subtle bugs where batch-level behavior is forgotten.
    // Subclasses can optionally delegate to listener (client app) by simply calling listener.onBatchComplete in implementation
    protected abstract suspend fun onBatchComplete(context: Context, batch: List<Output>)

}





