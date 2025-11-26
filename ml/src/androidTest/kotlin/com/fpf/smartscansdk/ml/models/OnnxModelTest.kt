package com.fpf.smartscansdk.ml.models


import android.content.Context
import androidx.test.core.app.ApplicationProvider
import ai.onnxruntime.*
import com.fpf.smartscansdk.ml.models.loaders.IModelLoader
import io.mockk.Runs
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.verify
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import java.nio.FloatBuffer


class OnnxModelInstrumentedTest {

    private lateinit var loader: IModelLoader<ByteArray>
    private lateinit var model: OnnxModel
    private lateinit var session: OrtSession

    private val context: Context = ApplicationProvider.getApplicationContext()

    @Before
    fun setup() {
        loader = mockk()

        // Mock static OrtEnvironment before model creation
        mockkStatic(OrtEnvironment::class)
        val mockEnv = mockk<OrtEnvironment>(relaxed = true)
        every { OrtEnvironment.getEnvironment() } returns mockEnv

        // Mock session creation
        session = mockk(relaxed = true)

        every { mockEnv.createSession(any<ByteArray>()) } returns session

        // Construct the model
        model = OnnxModel(loader)

        // Ensure loader.load() is stubbed and explicitly load the model so session is set
        runBlocking {
            coEvery { loader.load() } returns "fake_model".toByteArray()
            model.loadModel()
        }
    }


    @After
    fun teardown() {
        try {
            model.close()
        } catch (_: Exception) {
            // ignore
        }
        unmockkAll()
    }

    @Test
    fun `isLoaded returns true when session is set`() {
        assertTrue(model.isLoaded())
    }

    @Test
    fun `run returns mapped output`() = runBlocking {
        // Prepare a fake FloatBuffer for input
        val fakeBuffer = FloatBuffer.allocate(1)
        fakeBuffer.put(1.0f)
        fakeBuffer.flip()

        val tensorData = TensorData.FloatBufferTensor(fakeBuffer, longArrayOf(1))

        // Mock OnnxTensor creation
        val onnxTensor = mockk<OnnxTensor>(relaxed = true)
        mockkStatic(OnnxTensor::class)
        every { OnnxTensor.createTensor(any<OrtEnvironment>(), any<FloatBuffer>(), any<LongArray>()) } returns onnxTensor

        // Mock OnnxValue for result
        val onnxValue = mockk<OnnxValue>()
        every { onnxValue.value } returns floatArrayOf(1.0f)

        // Prepare a mutable map entry and mutable iterator
        val resultMap = mutableMapOf("out" to onnxValue)
        val result = mockk<OrtSession.Result>()
        every { result.iterator() } returns resultMap.entries.iterator()
        every { result.close() } just Runs

        // Mock session.run() to return the mocked result
        every { session.run(any<Map<String, OnnxTensor>>()) } returns result

        // Run the model
        val inputs = mapOf("in" to tensorData)
        val output = model.run(inputs)

        // Assertions
        assertTrue(output.containsKey("out"))
        assertEquals(floatArrayOf(1.0f).toList(), (output["out"] as FloatArray).toList())
    }


    @Test
    fun `getInputNames returns session input names`() {
        // Use a LinkedHashSet to keep deterministic order
        every { session.inputNames } returns linkedSetOf("input1", "input2")

        val inputNames = model.getInputNames()
        assertEquals(listOf("input1", "input2"), inputNames)
    }

    @Test
    fun `close closes session`() {
        model.close()
        // allow at least one close call since teardown may also close
        verify(atLeast = 1) { session.close() }
        assertFalse(model.isLoaded())
    }

    @Test
    fun `loadModel loads bytes and creates session`() = runBlocking {
        val bytes = "fake_model".toByteArray()
        coEvery { loader.load() } returns bytes

        // Use real loadModel which will call env.createSession(bytes)
        model.loadModel()

        assertTrue(model.isLoaded())
    }
}
