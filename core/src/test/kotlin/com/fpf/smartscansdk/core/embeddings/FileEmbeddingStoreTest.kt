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

    private fun getEmbedStoreFile(quantize: Boolean): File{
        val fileName = if(quantize) "embeddings_quant.bin" else "embeddings.bin"
        return File(tempDir, fileName)
    }

    private fun createStore(quantize: Boolean) = FileEmbeddingStore(getEmbedStoreFile(quantize), embeddingLength, quantize = quantize)

    fun randomEmbedding(quantize: Boolean): Embedding {
        val floatArray = FloatArray(embeddingLength) { Random.nextFloat() * 2f - 1f }
        return if(quantize) Embedding.QInt8(floatArray.toQInt8()) else Embedding.F32(floatArray)
    }

    private fun embedding(id: Long, date: Long, values: Embedding) =
        StoredEmbedding(id, date, values)

    private suspend fun testAddAndLoad(quantize: Boolean) {
        val store = createStore(quantize)
        val embeddings = listOf(
            embedding(1, 100, randomEmbedding(quantize)),
            embedding(2, 200, randomEmbedding(quantize))
        )

        store.add(embeddings)
        val loaded = store.get()

        assertEquals(2, loaded.size)
        assertEquals(embeddings[0].id, loaded[0].id)
        when (val a = embeddings[1].embedding) {
            is Embedding.F32 -> {
                val b = loaded[1].embedding as Embedding.F32
                assertEquals(a.vector.toList(), b.vector.toList())
            }
            is Embedding.QInt8 -> {
                val b = loaded[1].embedding as Embedding.QInt8
                assertEquals(a.vector.toList(), b.vector.toList())
            }
        }
    }

    private suspend fun testRemove(quantize: Boolean)  {
        val store = createStore(quantize)
        val embeddings = listOf(
            embedding(1L, 100, randomEmbedding(quantize)),
            embedding(2L, 200, randomEmbedding(quantize)),
            embedding(3L, 300, randomEmbedding(quantize))
        )

        store.add(embeddings)
        store.remove(listOf(2L))

        val remaining = store.get()
        Assertions.assertEquals(2, remaining.size)
        Assertions.assertFalse(remaining.any { it.id == 2L })
    }

    private suspend fun testCacheAndReload(quantize: Boolean)  {
        val store = createStore(quantize)
        val embeddings = listOf(embedding(1, 100, randomEmbedding(quantize)))
        store.add(embeddings)

        val firstLoad = store.get()
        store.clear()

        val secondLoad = store.get()
        assertTrue(firstLoad.zip(secondLoad).all { (a, b) ->
            a.id == b.id && a.date == b.date && when {
                a.embedding is Embedding.F32 && b.embedding is Embedding.F32 ->
                    a.embedding.vector.contentEquals(b.embedding.vector)
                a.embedding is Embedding.QInt8 && b.embedding is Embedding.QInt8 ->
                    a.embedding.vector.contentEquals(b.embedding.vector)

                else -> false
            }
        })
    }


    private suspend fun testUpdate(quantize: Boolean) {
        val store = createStore(quantize)

        val original = listOf(
            embedding(1L, 100, randomEmbedding(quantize)),
            embedding(2L, 200, randomEmbedding(quantize))
        )

        store.add(original)
        val updatedEmbedding = randomEmbedding(quantize)
        val updated = listOf(
            embedding(1L, 999, updatedEmbedding),
            embedding(3L, 300, randomEmbedding(quantize)) // does not exist
        )

        val updatedCount = store.update(updated)

        assertEquals(1, updatedCount)

        val loaded = store.get()

        assertEquals(2, loaded.size)

        val updatedEntry = loaded.first { it.id == 1L }
        assertEquals(999L, updatedEntry.date)
        assertTrue(
            when {
                updatedEntry.embedding is Embedding.F32 && updatedEmbedding is Embedding.F32 ->
                    updatedEntry.embedding.vector.contentEquals(updatedEmbedding.vector)

                updatedEntry.embedding is Embedding.QInt8 && updatedEmbedding is Embedding.QInt8 ->
                    updatedEntry.embedding.vector.contentEquals(updatedEmbedding.vector)

                else -> false
            }
        )

        val unchangedEntry = loaded.first { it.id == 2L }
        assertEquals(200L, unchangedEntry.date)
    }



    private suspend fun testAddRemovePersistence(quantize: Boolean){
        val store = createStore(quantize)

        val firstBatch = listOf(
            embedding(1L, 100, randomEmbedding(quantize)),
            embedding(2L, 200, randomEmbedding(quantize))
        )

        val secondBatch = listOf(
            embedding(3L, 300, randomEmbedding(quantize)),
            embedding(4L, 400, randomEmbedding(quantize))
        )

        store.add(firstBatch)

        // simulate fresh init state without cache
        store.clear()
        store.add(secondBatch)

        // remove one item after cache was reset
        store.remove(listOf(3L))
        store.save()

        val result = store.get()

        // Expected: first batch must still exist + second batch minus removed item
        assertEquals(3, result.size)
        assertTrue(result.any { it.id == 1L })
        assertTrue(result.any { it.id == 2L })
        assertTrue(result.any { it.id == 4L })
        assertTrue(result.none { it.id == 3L })
    }

    private suspend fun testConcurrentAdds( quantize: Boolean) = runTest {
        val store = createStore(quantize)

        val items = List(100) { i ->
            embedding(
                (i + 1).toLong(),
                ((i + 1) * 100).toLong(),
                randomEmbedding(quantize)
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

    @Test
    fun `add and load embeddings round trip`() = runTest {
        testAddAndLoad(quantize = false)
        testAddAndLoad(quantize = true)
    }

    @Test
    fun `update modifies existing embeddings and persists changes`() = runTest {
        testUpdate(quantize = false)
        testUpdate(quantize = true)
    }

    @Test
    fun `remove embeddings deletes specified ids`() = runTest {
        testRemove(quantize = false)
        testRemove(quantize = true)
    }

    @Test
    fun `cache is cleared and reloads`() = runTest {
        testCacheAndReload(quantize = false)
        testCacheAndReload(quantize = true)
    }

    @Test
    fun `adding embedding with wrong length throws`() = runTest {
        val store = createStore(quantize = false)
        val bad = listOf(embedding(1, 100, Embedding.F32(floatArrayOf(1f, 2f)))) // too short
        assertFailsWith<SmartScanException.InvalidEmbeddingDimension> {
            store.add(bad)
        }
    }

    @Test
    fun `corrupt header causes IOException`() = runTest {
        val store = createStore(false)
        val embeddings = listOf(embedding(1, 100, randomEmbedding(false)))
        store.add(embeddings)

        // corrupt first 4 bytes (count header)
        RandomAccessFile(getEmbedStoreFile(false), "rw").use { raf ->
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(Int.MAX_VALUE) // absurdly large
            buf.flip()
            raf.channel.position(0)
            raf.channel.write(buf)
        }

        assertFailsWith<SmartScanException.CorruptedEmbeddingStoreFile> {
            store.add(listOf(embedding(2, 200, randomEmbedding(false))))
        }
    }

    @Test
    fun `duplicated ids are not persisted to file`() = runTest {
        val store = createStore(false)
        val file = File(tempDir, "embeddings.bin")

        val first = embedding(1L, 100, randomEmbedding(false))
        store.add(listOf(first))

        val duplicate = embedding(1L, 200, randomEmbedding(false))
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
        val store = createStore(false)
        val first = embedding(1L, 100, randomEmbedding(false))
        store.add(listOf(first))

        val store2 = createStore(false)

        val loaded = store2.get(listOf(1L))
        assertEquals(1, loaded.size)
        assertEquals(1L, loaded[0].id)
    }


    @Test
    fun `add-remove sequence preserves full persisted state`() = runTest {
        testAddRemovePersistence(quantize = false)
        testAddRemovePersistence(quantize = true)
    }

    @Test
    fun `concurrent add calls work`() = runTest {
        testConcurrentAdds(quantize = false)
        testConcurrentAdds(quantize = true)
    }
}