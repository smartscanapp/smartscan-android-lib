package com.fpf.smartscansdk.core.embeddings

import android.util.Log
import com.fpf.smartscansdk.core.SmartScanException
import io.mockk.every
import io.mockk.mockkStatic
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue


@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class FileEmbeddingStoreTest {

    @BeforeEach
    fun setup() {
        mockkStatic(Log::class)
        every { Log.d(any<String>(), any<String>()) } returns 0
        every { Log.e(any<String>(), any<String>()) } returns 0
        every { Log.i(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
    }

    @TempDir
    lateinit var tempDir: File

    private val embeddingLength = 512

    private fun createStore(file: File = File(tempDir, "embeddings.bin")) =
        FileEmbeddingStore(file, embeddingLength)


    fun randomEmbedding(): FloatArray {
        return FloatArray(embeddingLength) { Random.nextFloat() * 2f - 1f }
    }
    private fun embedding(id: Long, date: Long, values: FloatArray) =
        StoredEmbedding(id, date, values)

    @Test
    fun `add and load embeddings round trip`() = runTest {
        val store = createStore()
        val embeddings = listOf(
            embedding(1, 100, randomEmbedding()),
            embedding(2, 200, randomEmbedding())
        )

        store.add(embeddings)
        val loaded = store.get()

        Assertions.assertEquals(2, loaded.size)
        Assertions.assertEquals(embeddings[0].id, loaded[0].id)
        Assertions.assertEquals(embeddings[1].embedding.toList(), loaded[1].embedding.toList())
    }

    @Test
    fun `add embeddings appends correctly`() = runTest {
        val store = createStore()
        val first = listOf(embedding(1, 100, randomEmbedding()))
        val second = listOf(embedding(2, 200, randomEmbedding()))

        store.add(first)
        store.add(second)

        val all = store.get()
        Assertions.assertEquals(2, all.size)
        Assertions.assertEquals(1, all[0].id)
        Assertions.assertEquals(2, all[1].id)
    }

    @Test
    fun `remove embeddings deletes specified ids`() = runTest {
        val store = createStore()
        val embeddings = listOf(
            embedding(1L, 100, randomEmbedding()),
            embedding(2L, 200, randomEmbedding()),
            embedding(3L, 300, randomEmbedding())
        )

        store.add(embeddings)
        store.remove(listOf(2L))

        val remaining = store.get()
        Assertions.assertEquals(2, remaining.size)
        Assertions.assertFalse(remaining.any { it.id == 2L })
    }

    @Test
    fun `cache is cleared and reloads`() = runTest {
        val store = createStore()
        val embeddings = listOf(embedding(1, 100, FloatArray(embeddingLength) { 0.1f }))
        store.add(embeddings)

        val firstLoad = store.get()
        store.clear()

        val secondLoad = store.get()
        assertTrue(firstLoad.zip(secondLoad).all { (a, b) ->
            a.id == b.id && a.date == b.date && a.embedding.contentEquals(b.embedding)
        })
    }

    @Test
    fun `adding embedding with wrong length throws`() = runTest {
        val store = createStore()
        val bad = listOf(embedding(1, 100, floatArrayOf(1f, 2f))) // too short

        assertFailsWith<SmartScanException.InvalidEmbeddingDimension> {
            store.add(bad)
        }
    }

    @Test
    fun `corrupt header causes IOException`() = runTest {
        val store = createStore()
        val embeddings = listOf(embedding(1, 100, randomEmbedding()))
        store.add(embeddings)

        // corrupt first 4 bytes (count header)
        RandomAccessFile(File(tempDir, "embeddings.bin"), "rw").use { raf ->
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(Int.MAX_VALUE) // absurdly large
            buf.flip()
            raf.channel.position(0)
            raf.channel.write(buf)
        }

        assertFailsWith<SmartScanException.CorruptedEmbeddingStoreFile> {
            store.add(listOf(embedding(2, 200, randomEmbedding())))
        }
    }

    @Test
    fun `duplicated ids are not persisted to file`() = runTest {
        val store = createStore()
        val file = File(tempDir, "embeddings.bin")

        val first = embedding(1L, 100, randomEmbedding())
        store.add(listOf(first))

        val duplicate = embedding(1L, 200, randomEmbedding())
        store.add(listOf(duplicate))

        // cache should show a single entry (overwritten by the second add)
        val loaded = store.get()
        assertEquals(1, loaded.size)
        assertEquals(1L, loaded[0].id)

        // Inspect raw file contents to count occurrences of id=1
        RandomAccessFile(file, "r").use { raf ->
            val ch = raf.channel
            val buffer = ch.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, ch.size())
                .order(ByteOrder.LITTLE_ENDIAN)

            val headerCount = buffer.int
            var occurrences = 0
            repeat(headerCount) {
                val id = buffer.long
                val date = buffer.long
                val floats = FloatArray(embeddingLength)
                val fb = buffer.asFloatBuffer()
                fb.get(floats)
                buffer.position(buffer.position() + embeddingLength * 4)
                if (id == 1L) occurrences++
            }

            Assertions.assertEquals(1, headerCount)
            Assertions.assertEquals(1, occurrences)
        }
    }

    @Test
    fun `get(ids) loads file if cache is empty`() = runTest {
        val store = createStore()
        val first = embedding(1L, 100, randomEmbedding())
        store.add(listOf(first))

        val store2 = createStore()

        val loaded = store2.get(listOf(1L))
        assertEquals(1, loaded.size)
        assertEquals(1L, loaded[0].id)
    }

    @Test
    fun `update modifies existing embeddings and persists changes`() = runTest {
        val store = createStore()

        val original = listOf(
            embedding(1L, 100, randomEmbedding()),
            embedding(2L, 200, randomEmbedding())
        )

        store.add(original)

        // Sanity check: assertion was correct after add here
//        val originalResult = store.get()
//
//        assertEquals(2, originalResult.size)

        val updatedEmbedding = randomEmbedding()
        val updated = listOf(
            embedding(1L, 999, updatedEmbedding),
            embedding(3L, 300, randomEmbedding()) // does not exist
        )

        val updatedCount = store.update(updated)

        assertEquals(1, updatedCount)

        val loaded = store.get()

        assertEquals(2, loaded.size)

        val updatedEntry = loaded.first { it.id == 1L }
        assertEquals(999L, updatedEntry.date)
        assertTrue(updatedEntry.embedding.contentEquals(updatedEmbedding))

        val unchangedEntry = loaded.first { it.id == 2L }
        assertEquals(200L, unchangedEntry.date)
    }

    @Test
    fun `add-remove sequence preserves full persisted state`() = runTest {
        val store = createStore()

        val firstBatch = listOf(
            embedding(1L, 100, randomEmbedding()),
            embedding(2L, 200, randomEmbedding())
        )

        val secondBatch = listOf(
            embedding(3L, 300, randomEmbedding()),
            embedding(4L, 400, randomEmbedding())
        )

        store.add(firstBatch)

        // simulate fresh init state without cache
        store.clear()

        store.add(secondBatch)

        // remove one item after cache was reset
        store.remove(listOf(3L))

        val result = store.get()

        // Expected: first batch must still exist + second batch minus removed item
        assertEquals(3, result.size)

        assertTrue(result.any { it.id == 1L })
        assertTrue(result.any { it.id == 2L })
        assertTrue(result.any { it.id == 4L })
        assertTrue(result.none { it.id == 3L })
    }

    @Test
    fun `concurrent remove calls work`() = runTest {
        val store = createStore()

        val batch = List(100) { i ->
            embedding(
                (i + 1).toLong(),
                ((i + 1) * 100).toLong(),
                randomEmbedding()
            )
        }

        store.add(batch)

        val nRemove = batch.size / 10


        val jobs = batch.take(nRemove).map { item ->
            launch(Dispatchers.IO) {
                store.remove(listOf(item.id))
            }
        }

        jobs.joinAll()

        val result = store.get()

        assertEquals(batch.size - nRemove, result.size)

        val removedIds = (1..nRemove).map { it.toLong() }.toSet()
        val expectedRemainingIds = batch.map { it.id }.toSet() - removedIds
        val remainingIds = result.map { it.id }.toSet()

        assertEquals(expectedRemainingIds, remainingIds)
    }

    @Test
    fun `concurrent add calls work`() = runTest {
        val store = createStore()

        val items = List(100) { i ->
            embedding(
                (i + 1).toLong(),
                ((i + 1) * 100).toLong(),
                randomEmbedding()
            )
        }

        val jobs = items.chunked(10).map { chunk ->
            launch(Dispatchers.IO) {
                store.add(chunk)
            }
        }

        jobs.joinAll()

        val result = store.get()

        assertEquals(items.size, result.size)
    }
}