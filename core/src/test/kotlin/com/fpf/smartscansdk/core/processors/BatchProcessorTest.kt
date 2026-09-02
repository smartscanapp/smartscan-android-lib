package com.fpf.smartscansdk.core.processors

import android.content.Context
import android.util.Log
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkConstructor
import io.mockk.mockkStatic
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BatchProcessorTest {

    private lateinit var context: Context
    private lateinit var mockListener: ProcessorListener<Int>

    @BeforeEach
    fun setup() {
        context = mockk(relaxed = true)
        mockListener = mockk(relaxed = true)
        mockkStatic(Log::class)
        every { Log.d(any<String>(), any<String>()) } returns 0
        every { Log.e(any<String>(), any<String>()) } returns 0
        every { Log.i(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0

        // Mock ConcurrencyController constructor to avoid real memory checks
        mockkConstructor(ConcurrencyController::class)
        every { anyConstructed<ConcurrencyController>().calculateConcurrency() } returns 2
    }

    // Simple concrete subclass for testing
    class TestProcessor(
        listener: ProcessorListener<Int>,
        concurrency: Concurrency,
        private val failOn: Set<Int> = emptySet(),
        batchSize: Int = 2
    ) : BatchProcessor<Int, Int>( listener, concurrency, batchSize) {

        override suspend fun onProcess(item: Int): Int {
            if (item in failOn) throw RuntimeException("Failed item $item")
            return item * 2
        }

        override suspend fun onBatchComplete(batch: List<Int>) {
            // no-op for testing
        }
    }

    @Test
    fun `run processes all items successfully`() = runBlocking {
        val concurrencyController = ConcurrencyController(context)
        val concurrency = Concurrency.Dynamic{concurrencyController.calculateConcurrency()}
        val processor = TestProcessor(mockListener, concurrency)
        val items = listOf(1, 2, 3, 4)

        val metrics = processor.run(items)

        assertTrue(metrics is ProcessorResult.Success)
        assertEquals(4, metrics.totalProcessed)

        coVerify { mockListener.onActive() }
        coVerify { mockListener.onProgress( match { it in 0f..1f }) }
        coVerify { mockListener.onComplete( any()) }
        coVerify(exactly = 0) { mockListener.onError(any(), any()) }
    }

    @Test
    fun `run handles empty input`() = runBlocking {
        val concurrencyController = ConcurrencyController(context)
        val concurrency = Concurrency.Dynamic{concurrencyController.calculateConcurrency()}
        val processor = TestProcessor(mockListener, concurrency)
        val items = emptyList<Int>()

        val metrics = processor.run(items)

        assertTrue(metrics is ProcessorResult.Success)
        assertEquals(0, metrics.totalProcessed)

        coVerify(exactly = 0) { mockListener.onProgress( any()) }
        coVerify(exactly = 1) { mockListener.onComplete( any()) }
    }

    @Test
    fun `run handles exceptions gracefully`() = runBlocking {
        val concurrencyController = ConcurrencyController(context)
        val concurrency = Concurrency.Dynamic{concurrencyController.calculateConcurrency()}
        val processor = TestProcessor(mockListener, concurrency,  failOn = setOf(2))
        val items = listOf(1, 2, 3)

        val metrics = processor.run(items)

        assertTrue(metrics is ProcessorResult.Success)
        assertEquals(2, metrics.totalProcessed)

        coVerify {
            mockListener.onError(
                match { it.message?.contains("Failed item 2") == true },
                2
            )
        }
    }
}