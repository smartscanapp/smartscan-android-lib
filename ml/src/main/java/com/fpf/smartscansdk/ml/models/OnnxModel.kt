package com.fpf.smartscansdk.ml.models

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.fpf.smartscansdk.ml.models.loaders.IModelLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext

class OnnxModel(override val loader: IModelLoader<ByteArray>) : BaseModel<TensorData>() {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    override suspend fun loadModel() = coroutineScope {
        withContext(Dispatchers.IO) {
            val bytes = loader.load()
            session = env.createSession(bytes)
        }
    }

    override fun isLoaded(): Boolean = session != null

    override fun run(inputs: Map<String, TensorData>): Map<String, Any> {
        val s = session ?: throw IllegalStateException("Model not loaded")
        val createdTensors: Map<String, OnnxTensor> = inputs.mapValues { (_, tensorData) ->
            createOnnxTensor(tensorData)
        }

        try {
            s.run(createdTensors).use { results ->
                return results.associate { it.key to it.value.value }
            }
        } finally {
            createdTensors.values.forEach { try { it.close() } catch (ignored: Exception) {} }
        }
    }

    private fun createOnnxTensor(tensorData: TensorData): OnnxTensor {
        return when (tensorData) {
            is TensorData.FloatBufferTensor -> OnnxTensor.createTensor(env, tensorData.data, tensorData.shape)
            is TensorData.IntBufferTensor -> OnnxTensor.createTensor(env, tensorData.data, tensorData.shape)
            is TensorData.LongBufferTensor -> OnnxTensor.createTensor(env, tensorData.data, tensorData.shape)
            is TensorData.DoubleBufferTensor -> OnnxTensor.createTensor(env, tensorData.data, tensorData.shape)
            is TensorData.ShortBufferTensor -> {
                val type = tensorData.type
                if (type != null) {
                    OnnxTensor.createTensor(env, tensorData.data, tensorData.shape, type)
                } else {
                    OnnxTensor.createTensor(env, tensorData.data, tensorData.shape)
                }
            }
            is TensorData.ByteBufferTensor -> OnnxTensor.createTensor(env, tensorData.data, tensorData.shape, tensorData.type)
        }
    }

    fun getInputNames(): List<String>? = session?.inputNames?.toList()

    fun getEnv(): OrtEnvironment = env

    override fun close() {
        session?.close()
        session = null
    }
}
