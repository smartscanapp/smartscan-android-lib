package com.fpf.smartscansdk.ml.models.providers.embeddings.clip

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import com.fpf.smartscansdk.ml.data.ResourceId
import com.fpf.smartscansdk.ml.data.TensorData
import com.fpf.smartscansdk.ml.models.providers.embeddings.clip.ClipConfig.IMAGE_SIZE_X
import com.fpf.smartscansdk.ml.models.providers.embeddings.clip.ClipConfig.IMAGE_SIZE_Y
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.providers.embeddings.clip.ClipImageEmbedder
import io.mockk.*
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.verify
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import java.nio.FloatBuffer


class ClipImageEmbedderInstrumentedTest {

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
        val embedder = ClipImageEmbedder(context, ResourceId(0))

        // replace private model with a mock
        val mockModel = mockk<OnnxModel>(relaxed = true)
        coEvery { mockModel.loadModel() } answers {
            every { mockModel.isLoaded() } returns true
        }
        val modelField = embedder::class.java.getDeclaredField("model")
        modelField.isAccessible = true
        modelField.set(embedder, mockModel)

        embedder.initialize()

        coVerify { mockModel.loadModel() }
        assertTrue(embedder.isInitialized())
    }

    @Test
    fun `embed returns normalized vector of expected dimension`() = runBlocking {
        val embedder = ClipImageEmbedder(context, ResourceId(0))

        // mock internal model
        val mockModel = mockk<OnnxModel>(relaxed = true)
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns mockk<OrtEnvironment>()

        // prepare fake output: Array<FloatArray> with length 512
        val raw = Array(1) { FloatArray(embedder.embeddingDim) { 1.0f } }
        every { mockModel.run(any<Map<String, TensorData>>()) } returns mapOf("out" to raw)

        // mock tensor creation and closing; specify types so mockk can infer overload
        val mockTensor = mockk<OnnxTensor>(relaxed = true)
        every { OnnxTensor.createTensor(any<OrtEnvironment>(), any<FloatBuffer>(), any<LongArray>()) } returns mockTensor
        every { mockTensor.close() } just Runs

        // inject mockModel
        val modelField = embedder::class.java.getDeclaredField("model")
        modelField.isAccessible = true
        modelField.set(embedder, mockModel)

        val bmp = Bitmap.createBitmap(IMAGE_SIZE_X, IMAGE_SIZE_Y, Bitmap.Config.ARGB_8888)

        val embedding = embedder.embed(bmp)

        assertEquals(embedder.embeddingDim, embedding.size)
        // L2 norm should be ~1.0 after normalizeL2
        val l2 = sqrt(embedding.map { it * it }.sum())
        assertTrue(abs(l2 - 1.0f) < 1e-3)
    }

    @Test
    fun `embedBatch returns embeddings for all items`() = runBlocking {
        val embedder = ClipImageEmbedder(context, ResourceId(0))

        val mockModel = mockk<OnnxModel>(relaxed = true)
        every { mockModel.isLoaded() } returns true
        every { mockModel.getInputNames() } returns listOf("input")
        every { mockModel.getEnv() } returns mockk<OrtEnvironment>()

        val raw = Array(1) { FloatArray(embedder.embeddingDim) { 1.0f } }
        every { mockModel.run(any<Map<String, TensorData>>()) } returns mapOf("out" to raw)

        val mockTensor = mockk<OnnxTensor>(relaxed = true)
        every { OnnxTensor.createTensor(any<OrtEnvironment>(), any<FloatBuffer>(), any<LongArray>()) } returns mockTensor
        every { mockTensor.close() } just Runs

        val modelField = embedder::class.java.getDeclaredField("model")
        modelField.isAccessible = true
        modelField.set(embedder, mockModel)

        val bmp1 = Bitmap.createBitmap(IMAGE_SIZE_X, IMAGE_SIZE_Y, Bitmap.Config.ARGB_8888)
        val bmp2 = Bitmap.createBitmap(IMAGE_SIZE_X, IMAGE_SIZE_Y, Bitmap.Config.ARGB_8888)

        val results = embedder.embedBatch( listOf(bmp1, bmp2))

        assertEquals(2, results.size)
        assertEquals(embedder.embeddingDim, results[0].size)
    }

    @Test
    fun `closeSession closes model once`() {
        val embedder = ClipImageEmbedder(context, ResourceId(0))
        val mockModel = mockk<OnnxModel>(relaxed = true)
        val modelField = embedder::class.java.getDeclaredField("model")
        modelField.isAccessible = true
        modelField.set(embedder, mockModel)

        // ensure model implements AutoCloseable close()
        every { (mockModel as AutoCloseable).close() } just Runs

        embedder.closeSession()
        embedder.closeSession() // second call should be no-op

        verify(exactly = 1) { (mockModel as AutoCloseable).close() }
    }
}
