package com.fpf.smartscansdk.core.processors

import android.content.Context
import android.util.Log
import com.fpf.smartscansdk.core.data.Metrics
import com.fpf.smartscansdk.core.data.ProcessOptions
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkConstructor
import io.mockk.mockkStatic
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BatchProcessorTest {

    private lateinit var context: Context
    private lateinit var mockListener: IProcessorListener<Int, Int>

    @BeforeEach
    fun setup() {
        context = mockk(relaxed = true)
        mockListener = mockk(relaxed = true)
        mockkStatic(Log::class)
        every { Log.d(any<String>(), any<String>()) } returns 0
        every { Log.e(any<String>(), any<String>()) } returns 0
        every { Log.i(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0

        // Mock Memory constructor to avoid real memory checks
        mockkConstructor(Memory::class)
        every { anyConstructed<Memory>().calculateConcurrencyLevel() } returns 2
    }

    // Simple concrete subclass for testing
    class TestProcessor(
        context: Context,
        listener: IProcessorListener<Int, Int>,
        private val failOn: Set<Int> = emptySet(),
        options: ProcessOptions = ProcessOptions(batchSize = 2)
    ) : BatchProcessor<Int, Int>(context, listener, options) {

        override suspend fun onProcess(context: Context, item: Int): Int {
            if (item in failOn) throw RuntimeException("Failed item $item")
            return item * 2
        }

        override suspend fun onBatchComplete(context: Context, batch: List<Int>) {
            // no-op for testing
        }
    }

    @Test
    fun `run processes all items successfully`() = runBlocking {
        val processor = TestProcessor(context, mockListener)
        val items = listOf(1, 2, 3, 4)

        val metrics = processor.run(items)

        assertTrue(metrics is Metrics.Success)
        assertEquals(4, metrics.totalProcessed)

        coVerify { mockListener.onActive(context.applicationContext) }
        coVerify { mockListener.onProgress(context.applicationContext, match { it in 0f..1f }) }
        coVerify { mockListener.onComplete(context.applicationContext, any()) }
        coVerify(exactly = 0) { mockListener.onError(any(), any(), any()) }
    }

    @Test
    fun `run handles empty input`() = runBlocking {
        val processor = TestProcessor(context, mockListener)
        val items = emptyList<Int>()

        val metrics = processor.run(items)

        assertTrue(metrics is Metrics.Success)
        assertEquals(0, metrics.totalProcessed)

        coVerify(exactly = 0) { mockListener.onProgress(context, any()) }
        coVerify(exactly = 1) { mockListener.onComplete(context.applicationContext, any()) }
    }

    @Test
    fun `run handles item failures`() = runBlocking {
        val processor = TestProcessor(context, mockListener, failOn = setOf(2, 4))
        val items = listOf(1, 2, 3, 4)

        val metrics = processor.run(items)

        assertTrue(metrics is Metrics.Success) // failures are logged but do not abort
        assertEquals(2, metrics.totalProcessed) // only successful items counted

        verify {
            mockListener.onError(
                context.applicationContext,
                match { it.message?.contains("Failed item") == true },
                any()
            )
        }
    }

    @Test
    fun `run handles exceptions gracefully`() = runBlocking {
        val processor = TestProcessor(context, mockListener, failOn = setOf(2))
        val items = listOf(1, 2, 3)

        val metrics = processor.run(items)

        assertTrue(metrics is Metrics.Success)
        assertEquals(2, metrics.totalProcessed)

        coVerify {
            mockListener.onError(
                context.applicationContext,
                match { it.message?.contains("Failed item 2") == true },
                2
            )
        }
    }
}