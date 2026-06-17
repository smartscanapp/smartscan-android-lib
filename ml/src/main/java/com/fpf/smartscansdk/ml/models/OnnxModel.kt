package com.fpf.smartscansdk.ml.models

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.ml.models.loaders.ModelLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext

class OnnxModel(override val loader: ModelLoader<ByteArray>) : BaseModel<OnnxTensor, OnnxValue>() {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    override suspend fun loadModel() = coroutineScope {
        withContext(Dispatchers.IO) {
            val bytes = loader.load()
            session = env.createSession(bytes)
        }
    }

    override fun isLoaded(): Boolean = session != null

    override fun run(inputs: Map<String, OnnxTensor>): Map<String, OnnxValue> {
        val onnxSession = session ?: throw SmartScanException.ModelNotInitialised()

        try {
            val results = onnxSession.run(inputs)
            return results.associate { it.key to it.value }
        } finally {
            inputs.values.forEach { try { it.close() } catch (_: Exception) {} }
        }
    }

    fun getInputNames(): List<String>? = session?.inputNames?.toList()

    fun getOutputNames(): List<String>? = session?.outputNames?.toList()

    fun getEnv(): OrtEnvironment = env

    override fun close() {
        session?.close()
        session = null
    }
}
