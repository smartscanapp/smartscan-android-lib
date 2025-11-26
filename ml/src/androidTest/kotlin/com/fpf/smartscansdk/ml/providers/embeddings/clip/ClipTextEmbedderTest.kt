package com.fpf.smartscansdk.ml.providers.embeddings.clip

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.fpf.smartscansdk.ml.models.TensorData
import com.fpf.smartscansdk.ml.models.loaders.ResourceId
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.providers.embeddings.clip.ClipTextEmbedder
import io.mockk.*
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import java.nio.LongBuffer

class ClipTextEmbedderInstrumentedTest {
    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        mockkStatic(OnnxTensor::class)
    }

    @After
    fun teardown() {
        unmockkAll()
    }

    @Test
    fun `initialize calls model loadModel and sets initialized`() = runBlocking {
        val embedder = ClipTextEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        coEvery { mockModel.loadModel() } answers { every { mockModel.isLoaded() } returns true }
        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        embedder.initialize()

        coVerify { mockModel.loadModel() }
        assertTrue(embedder.isInitialized())
    }

    @Test
    fun `embed returns normalized vector of expected dimension`() = runBlocking {
        val embedder = ClipTextEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns mockk<OrtEnvironment>()

        val raw = Array(1) { FloatArray(embedder.embeddingDim) { 1.0f } }
        every { mockModel.run(any<Map<String, TensorData>>()) } returns mapOf("out" to raw)

        val mockTensor = mockk<OnnxTensor>(relaxed = true)
        every { OnnxTensor.createTensor(any<OrtEnvironment>(), any<LongBuffer>(), any<LongArray>()) } returns mockTensor
        every { mockTensor.close() } just Runs

        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        val embedding = embedder.embed("Hello world!")

        assertEquals(embedder.embeddingDim, embedding.size)
        val l2 = sqrt(embedding.map { it * it }.sum())
        assertTrue(abs(l2 - 1.0f) < 1e-3)
    }

    @Test
    fun `embedBatch returns embeddings for all items`() = runBlocking {
        val embedder = ClipTextEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns mockk<OrtEnvironment>()

        val raw = Array(1) { FloatArray(embedder.embeddingDim) { 1.0f } }
        every { mockModel.run(any<Map<String, TensorData>>()) } returns mapOf("out" to raw)

        val mockTensor = mockk<OnnxTensor>(relaxed = true)
        every { OnnxTensor.createTensor(any<OrtEnvironment>(), any<LongBuffer>(), any<LongArray>()) } returns mockTensor
        every { mockTensor.close() } just Runs

        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        val texts = listOf("Hello", "World")
        val results = embedder.embedBatch( texts)

        assertEquals(2, results.size)
        assertEquals(embedder.embeddingDim, results[0].size)
    }

    @Test
    fun `closeSession closes model once`() {
        val embedder = ClipTextEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        every { (mockModel as AutoCloseable).close() } just Runs

        embedder.closeSession()
        embedder.closeSession() // second call should be no-op

        verify(exactly = 1) { (mockModel as AutoCloseable).close() }
    }

    @Test
    fun `embed handles strings longer than 77 tokens`() = runBlocking {
        val embedder = ClipTextEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns mockk<OrtEnvironment>()

        val raw = Array(1) { FloatArray(embedder.embeddingDim) { 1.0f } }
        every { mockModel.run(any<Map<String, TensorData>>()) } returns mapOf("out" to raw)

        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        val longText = "a".repeat(2000)
        val embedding = embedder.embed(longText)

        assertEquals(embedder.embeddingDim, embedding.size)
        val l2 = sqrt(embedding.map { it * it }.sum())
        assertTrue(abs(l2 - 1.0f) < 1e-3)
    }

}
