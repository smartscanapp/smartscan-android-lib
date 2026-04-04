package com.fpf.smartscansdk.ml.providers.embeddings.clip

import android.app.Application
import android.content.Context
import androidx.annotation.RawRes
import com.fpf.smartscansdk.core.embeddings.TextEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.core.models.ModelManager
import com.fpf.smartscansdk.core.models.ModelName
import com.fpf.smartscansdk.core.models.ModelRegistry
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceOnnxLoader
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.ml.models.TensorData
import kotlinx.coroutines.*
import java.io.File
import java.nio.LongBuffer

// ResIds can be provided for bundled models
// If not provided the model registry paths are used (throws if model not downloaded)
class ClipTextEmbedder(
    private val context: Context,
    @RawRes modelResId: Int? = null,
    @RawRes vocabResId: Int? = null,
    @RawRes mergesResId: Int? = null,
    ) : TextEmbeddingProvider {

    private val model: OnnxModel = if (modelResId != null) {
        OnnxModel(ResourceOnnxLoader(context.resources, modelResId))
    } else {
        if (!ModelManager.modelExists(context, ModelName.CLIP_VIT_B_32_TEXT)) throw IllegalStateException("Model not downloaded")
        val modelInfo = ModelRegistry[ModelName.CLIP_VIT_B_32_TEXT]!!
        val modelDir = ModelManager.getModelFile(context, modelInfo = modelInfo)
        val modelFile = File(modelDir, modelInfo.resourceFiles!![0])
        OnnxModel(FileOnnxLoader(modelFile.absolutePath))
    }

    private val tokenizer = if (vocabResId != null && mergesResId != null) {
        ClipTokenizer.load(context, vocabResId, mergesResId)
    } else {
        val modelInfo = ModelRegistry[ModelName.CLIP_VIT_B_32_TEXT]!!
        val modelDir = ModelManager.getModelFile(context, modelInfo = modelInfo)
        val vocabFile = File(modelDir, modelInfo.resourceFiles!![1])
        val mergesFile = File(modelDir, modelInfo.resourceFiles!![2])
        ClipTokenizer.load(vocabFile, mergesFile)
    }
    private val tokenBOS = 49406
    private val tokenEOS = 49407

    override val embeddingDim: Int = 512
    override val maxTokens: Int = 77

    private var closed = false

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw IllegalStateException("Model not initialized")

        val clean = Regex("[^A-Za-z0-9 ]").replace(data, "").lowercase()
        var tokens = (mutableListOf(tokenBOS) + tokenizer.encode(clean) + tokenEOS).take(maxTokens).toMutableList()
        if (tokens.size < maxTokens) tokens += List(maxTokens - tokens.size) { 0 }


        val inputIds = LongBuffer.allocate(1 * maxTokens).apply {
            tokens.forEach { put(it.toLong()) }
            rewind()
        }
        val inputShape = longArrayOf(1, maxTokens.toLong())
        val inputName = model.getInputNames()?.firstOrNull() ?: throw IllegalStateException("Model inputs not available")
        val output = model.run(mapOf(inputName to TensorData.LongBufferTensor(inputIds, inputShape)))
        normalizeL2((output.values.first() as Array<FloatArray>)[0])
    }


    override suspend fun embedBatch(data: List<String>): List<FloatArray> {
        val allEmbeddings = mutableListOf<FloatArray>()

        val processor = object : BatchProcessor<String, FloatArray>(context = context.applicationContext as Application) {
            override suspend fun onProcess(context: Context, item: String): FloatArray {
                return embed(item)
            }
            override suspend fun onBatchComplete(context: Context, batch: List<FloatArray>) {
                allEmbeddings.addAll(batch)
            }
        }

        processor.run(data)
        return allEmbeddings
    }

    override fun closeSession() {
        if (closed) return
        closed = true
        (model as? AutoCloseable)?.close()
    }
}
