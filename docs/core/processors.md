# Processors Documentation

## MemoryOptions

Configuration for runtime memory-based concurrency scaling.

Fields:

* `lowMemoryThreshold: Long` — below this value, minimum concurrency is used
* `highMemoryThreshold: Long` — above this value, maximum concurrency is used
* `minConcurrency: Int` — lower bound for parallel execution
* `maxConcurrency: Int` — upper bound for parallel execution

---

## Memory

Utility for adaptive concurrency control based on available system memory.

Constructor:

```kotlin
Memory(context: Context, memoryOptions: MemoryOptions? = null)
```

---

### getFreeMemory(): Long

Returns available system memory using `ActivityManager.MemoryInfo`.

---

### calculateConcurrencyLevel(): Int

Computes concurrency level based on free memory.

Behavior:

* If memory < `lowMemoryThreshold` → returns `minConcurrency`
* If memory ≥ `highMemoryThreshold` → returns `maxConcurrency`
* Otherwise scales linearly between min and max

---

## Metrics

Sealed result type representing batch processing outcome.

### Success

```kotlin
Success(totalProcessed: Int, timeElapsed: Long)
```

Fields:

* `totalProcessed` — number of successfully processed items
* `timeElapsed` — total execution time in milliseconds

---

### Failure

```kotlin
Failure(processedBeforeFailure: Int, timeElapsed: Long, error: Exception)
```

Fields:

* `processedBeforeFailure` — items processed before failure
* `timeElapsed` — elapsed time before failure
* `error` — thrown exception

---

## ProcessorListener<Input, Output>

Event callbacks for batch processing lifecycle.

Methods:

* `onActive(context)` — called before processing starts
* `onBatchComplete(context, batch)` — called after each batch
* `onComplete(context, metrics)` — called on successful completion
* `onProgress(context, progress)` — progress updates (0.0–1.0)
* `onError(context, error, item)` — per-item failure handler
* `onFail(context, failureMetrics)` — fatal failure callback

Notes:

* Most callbacks are `suspend` except `onError`
* Designed for external observability without affecting core logic

---

## BatchProcessor<Input, Output>

Abstract base class for concurrent batch processing.

Constructor:

```kotlin
BatchProcessor(
    context: Context,
    listener: ProcessorListener<Input, Output>? = null,
    memoryOptions: MemoryOptions = MemoryOptions(),
    batchSize: Int = 10
)
```

---

### run(items: List<Input>): Metrics

Main execution entry point.

Behavior:

* Splits input into batches (`batchSize`)
* Computes concurrency per batch using `Memory`
* Executes items concurrently using coroutine `async`
* Uses `Semaphore` to limit parallel execution
* Emits lifecycle events via `ProcessorListener`
* Tracks progress incrementally
* Returns either `Metrics.Success` or `Metrics.Failure`

Execution flow:

* Initialize timing and counters
* Trigger `onActive`
* For each batch:

    * Determine concurrency
    * Process items concurrently
    * Collect successful outputs
    * Call `onBatchComplete`
    * Update progress
* On completion:

    * Return success metrics
* On exception:

    * Return failure metrics

---

### onProcess(context, item): Output

Abstract method.

Defines per-item processing logic.

---

### onBatchComplete(context, batch)

Abstract method.

Called after each batch is processed successfully.

---

## Concurrency Model

* Batch-level splitting via `chunked(batchSize)`
* Item-level parallelism via coroutines
* Concurrency limited by:

    * system memory (via `Memory`)
    * semaphore per batch

---

## Error Handling

* Per-item errors:

    * caught inside coroutine
    * reported via `onError`
    * do not stop execution

* Global errors:

    * caught in `run`
    * returned as `Metrics.Failure`
    * reported via `onFail`

* Cancellation:

    * rethrown immediately (`CancellationException` preserved)

---

## Usage

### Implementing a processor

```kotlin
class MyProcessor(
    context: Context,
    listener: ProcessorListener<Input, Output>? = null
) : BatchProcessor<Input, Output>(context, listener) {

    override suspend fun onProcess(context: Context, item: Input): Output {
        return processItem(item)
    }

    override suspend fun onBatchComplete(context: Context, batch: List<Output>) {
        // batch-level handling
    }
}
```

---


## Design Notes

* Concurrency is dynamically adjusted based on system memory
* Batch processing reduces coroutine overhead for large inputs
* Listener is optional but recommended for observability
* Failures are isolated per item to maximize throughput
* Designed for long-running background workloads rather than UI-bound tasks
