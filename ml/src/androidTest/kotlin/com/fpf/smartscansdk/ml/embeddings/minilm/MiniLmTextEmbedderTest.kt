package com.fpf.smartscansdk.ml.embeddings.minilm

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.fpf.smartscansdk.core.embeddings.embedBatch
import com.fpf.smartscansdk.ml.embeddings.minilm.MiniLMTextEmbedder
import com.fpf.smartscansdk.ml.embeddings.minilm.MiniLmTokenizer
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import io.mockk.*
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MiniLmTextEmbedderTest {
    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        mockkStatic(OnnxTensor::class)
        mockkObject(MiniLmTokenizer.Companion)

        val mockTokenizer = mockk<MiniLmTokenizer>(relaxed = true)
        every { mockTokenizer.encode(any()) } returns Pair(intArrayOf(1,2,3), intArrayOf(1,1,1))
        every { MiniLmTokenizer.load(any(), any(), any()) } returns mockTokenizer
    }

    @After
    fun teardown() {
        unmockkAll()
    }

    @Test
    fun modelInitializationTest() = runBlocking {
        val embedder = MiniLMTextEmbedder(
            context,
            ModelAssetSource.Resource(0),
            ModelAssetSource.Resource(1),
            ModelAssetSource.Resource(2)
        )
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
    fun embeddingTest() = runBlocking {
        val embedder = MiniLMTextEmbedder(
            context,
            ModelAssetSource.Resource(0),
            ModelAssetSource.Resource(1),
            ModelAssetSource.Resource(2)
        )
        val mockModel = mockk<OnnxModel>(relaxed = true)
        val env = OrtEnvironment.getEnvironment()
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns env

        val raw = FloatArray(embedder.embeddingDim) { 1.0f }
        val outputTensor = OnnxTensor.createTensor(
            mockModel.getEnv(), // or real OrtEnvironment if available
            FloatBuffer.wrap(raw),
            longArrayOf(1, embedder.embeddingDim.toLong())
        )
        every { mockModel.run(any<Map<String, OnnxTensor>>()) } returns mapOf("out" to outputTensor)

        val mockTensor = mockk<OnnxTensor>(relaxed = true)
        every { OnnxTensor.createTensor(mockModel.getEnv(), any<LongBuffer>(), any<LongArray>()) } returns mockTensor
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
    fun batchEmbeddingTest() = runBlocking {
        val embedder = MiniLMTextEmbedder(
            context,
            ModelAssetSource.Resource(0),
            ModelAssetSource.Resource(1),
            ModelAssetSource.Resource(2)
        )

        val mockModel = mockk<OnnxModel>(relaxed = true)
        val env = OrtEnvironment.getEnvironment()

        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns env

        every { mockModel.run(any()) } answers {
            val raw = FloatArray(embedder.embeddingDim) { 1.0f }

            val tensor = OnnxTensor.createTensor(
                mockModel.getEnv(),
                FloatBuffer.wrap(raw),
                longArrayOf(1, embedder.embeddingDim.toLong())
            )

            mapOf("out" to tensor)
        }

        val field = embedder::class.java.getDeclaredField("model")
        field.isAccessible = true
        field.set(embedder, mockModel)

        val texts = listOf("Hello", "World")
        val results = embedBatch(
            context.applicationContext,
            embedder,
            texts
        )

        assertEquals(2, results.size)
        assertEquals(embedder.embeddingDim, results[0].size)
    }

    @Test
    fun maxTokenHandlingTest() = runBlocking {
        val embedder = MiniLMTextEmbedder(
            context,
            ModelAssetSource.Resource(0),
            ModelAssetSource.Resource(1),
            ModelAssetSource.Resource(2)
        )
        val mockModel = mockk<OnnxModel>(relaxed = true)
        val env = OrtEnvironment.getEnvironment()

        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns env

        val raw = FloatArray(embedder.embeddingDim) { 1.0f }
        val outputTensor = OnnxTensor.createTensor(
            mockModel.getEnv(), // or real OrtEnvironment if available
            FloatBuffer.wrap(raw),
            longArrayOf(1, embedder.embeddingDim.toLong())
        )
        every { mockModel.run(any<Map<String, OnnxTensor>>()) } returns mapOf("out" to outputTensor)

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
