package com.fpf.smartscansdk.core.processors

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicInteger

abstract class BatchProcessor<Input, Output>(
    protected val listener: ProcessorListener<Input>? = null,
    protected val concurrency: Concurrency = Concurrency.Fixed(1),
    protected val batchSize: Int = 10
) {

    suspend fun run(items: List<Input>): ProcessorResult = withContext(Dispatchers.IO) {
        val processedCount = AtomicInteger(0)
        val startTime = System.currentTimeMillis()
        var totalSuccess = 0

        try {
            if (items.isEmpty()) {
                val processorResult = ProcessorResult.Success()
                listener?.onComplete(processorResult)
                return@withContext processorResult
            }
            listener?.onActive()

            for (batch in items.chunked(batchSize)) {
                val currentConcurrency = when(concurrency){
                    is Concurrency.Fixed -> concurrency.concurrency
                    is Concurrency.Dynamic -> concurrency.calculateConcurrency()
                }
                val semaphore = Semaphore(currentConcurrency)

                val deferredResults = batch.map { item ->
                    async {
                        semaphore.withPermit {
                            try {
                                val output = onProcess(item)
                                item to Result.success(output)
                            } catch (e: Exception) {
                                item to Result.failure(e)
                            }finally {
                                val current = processedCount.incrementAndGet()
                                val progress = current.toFloat() / items.size
                                listener?.onProgress(progress)
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
                            listener?.onError(error, result.first)
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
                onBatchComplete(successfulResults)
            }

            val endTime = System.currentTimeMillis()
            val processorResult = ProcessorResult.Success(totalSuccess, timeElapsed = endTime - startTime)

            listener?.onComplete(processorResult)
            processorResult
        }
        catch (e: CancellationException) {
            listener?.onCancel()
            throw e
        }
        catch (e: Exception) {
            val processorResult = ProcessorResult.Failure(
                totalProcessed = totalSuccess,
                timeElapsed = System.currentTimeMillis() - startTime,
                error = e
            )
            listener?.onFail(processorResult)
            processorResult
        }
    }

    protected abstract suspend fun onProcess(item: Input): Output

    protected abstract suspend fun onBatchComplete(batch: List<Output>)

}

