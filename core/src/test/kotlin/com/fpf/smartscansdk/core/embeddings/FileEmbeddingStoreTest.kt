package com.fpf.smartscansdk.core.embeddings

import android.util.Log
import io.mockk.every
import io.mockk.mockkStatic
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue


@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class FileEmbeddingStoreTest {

    @BeforeEach
    fun setup() {
        mockkStatic(android.util.Log::class)
        every { Log.d(any<String>(), any<String>()) } returns 0
        every { Log.e(any<String>(), any<String>()) } returns 0
        every { Log.i(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
    }

    @TempDir
    lateinit var tempDir: File

    private val embeddingLength = 4

    private fun createStore(file: File = File(tempDir, "embeddings.bin")) =
        FileEmbeddingStore(file, embeddingLength)

    private fun embedding(id: Long, date: Long, values: FloatArray) =
        StoredEmbedding(id, date, values)

    @Test
    fun `add and load embeddings round trip`() = runTest {
        val store = createStore()
        val embeddings = listOf(
            embedding(1, 100, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)),
            embedding(2, 200, floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f))
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
        val first = listOf(embedding(1, 100, floatArrayOf(1f, 2f, 3f, 4f)))
        val second = listOf(embedding(2, 200, floatArrayOf(5f, 6f, 7f, 8f)))

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
            embedding(1L, 100, floatArrayOf(1f, 1f, 1f, 1f)),
            embedding(2L, 200, floatArrayOf(2f, 2f, 2f, 2f)),
            embedding(3L, 300, floatArrayOf(3f, 3f, 3f, 3f))
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

        assertFailsWith<IllegalArgumentException> {
            store.add(bad)
        }
    }

    @Test
    fun `corrupt header causes IOException`() = runTest {
        val store = createStore()
        val embeddings = listOf(embedding(1, 100, FloatArray(embeddingLength) { 1f }))
        store.add(embeddings)

        // corrupt first 4 bytes (count header)
        RandomAccessFile(File(tempDir, "embeddings.bin"), "rw").use { raf ->
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(Int.MAX_VALUE) // absurdly large
            buf.flip()
            raf.channel.position(0)
            raf.channel.write(buf)
        }

        assertFailsWith<IOException> {
            store.add(listOf(embedding(2, 200, FloatArray(embeddingLength) { 2f })))
        }
    }

    @Test
    fun `query batch retrieval with start and end works`() = runTest {
        val store = createStore()

        val embeddings = listOf(
            embedding(1, 100, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)),
            embedding(2, 200, floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f)),
            embedding(3, 300, floatArrayOf(0.9f, 1.0f, 1.1f, 1.2f))
        )
        store.add(embeddings)

        // trigger initial query to populate cachedIds
        store.query(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f), topK = 3, threshold = 0f)

        // fetch first two cached embeddings (order-agnostic)
        val batch1 = store.query(0, 2)
        assertEquals(2, batch1.size)
        Assertions.assertTrue(batch1.map { it }.all { it in listOf(1L, 2L, 3L) })

        // fetch last cached embedding
        val batch2 = store.query(2, 3)
        assertEquals(1, batch2.size)
        Assertions.assertTrue(batch2[0] in listOf(1L, 2L, 3L))

        // out-of-bounds requests return empty
        val batch3 = store.query(3, 5)
        assertEquals(0, batch3.size)
    }

    @Test
    fun `duplicated ids are not persisted to file`() = runTest {
        val store = createStore()
        val file = File(tempDir, "embeddings.bin")

        val first = embedding(1L, 100, floatArrayOf(1f, 2f, 3f, 4f))
        store.add(listOf(first))

        val duplicate = embedding(1L, 200, floatArrayOf(5f, 6f, 7f, 8f))
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
        val first = embedding(1L, 100, floatArrayOf(1f, 2f, 3f, 4f))
        store.add(listOf(first))

        val store2 = createStore()

        val loaded = store2.get(listOf(1L))
        assertEquals(1, loaded.size)
        assertEquals(1L, loaded[0].id)
    }


}