package com.fpf.smartscansdk.ml.providers.embeddings.minilm

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
import com.fpf.smartscansdk.ml.models.TensorData
import kotlinx.coroutines.*
import java.nio.LongBuffer
import com.fpf.smartscansdk.core.processors.BatchProcessor
import java.io.File

class MiniLMTextEmbedder(
    private val context: Context,
    @RawRes modelResId: Int? = null,
    @RawRes vocabResId: Int? = null,
    @RawRes mergesResId: Int? = null,
    override val maxTokens: Int, // required as param because sentence transformer models can be exported with different token lengths
    ) : TextEmbeddingProvider {
    private val model: OnnxModel = if (modelResId != null) {
        OnnxModel(ResourceOnnxLoader(context.resources, modelResId))
    } else {
        if (!ModelManager.modelExists(context, ModelName.ALL_MINILM_L6_V2)) throw IllegalStateException("Model not downloaded")
        val modelInfo = ModelRegistry[ModelName.ALL_MINILM_L6_V2]!!
        val modelDir = ModelManager.getModelFile(context, modelInfo = modelInfo)
        val modelFile = File(modelDir, modelInfo.resourceFiles!![0])
        OnnxModel(FileOnnxLoader(modelFile.absolutePath))
    }

    private val tokenizer = if (vocabResId != null && mergesResId != null) {
        MiniLmTokenizer.load(context, vocabResId, mergesResId)
    } else {
        val modelInfo = ModelRegistry[ModelName.ALL_MINILM_L6_V2]!!
        val modelDir = ModelManager.getModelFile(context, modelInfo = modelInfo)
        val vocabFile = File(modelDir, modelInfo.resourceFiles!![1])
        val configFile = File(modelDir, modelInfo.resourceFiles!![2])
        MiniLmTokenizer.load(vocabFile, configFile)
    }

    private var closed = false
    override val embeddingDim: Int = 384

    override suspend fun initialize()  {
        model.loadModel()
    }

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw IllegalStateException("Model not initialized")

        val (inputIds, attentionMask) = tokenizer.encode(data)
        val inputShape = longArrayOf(1, maxTokens.toLong())
        val inputIdsTensor = TensorData.LongBufferTensor(LongBuffer.wrap(inputIds), inputShape)
        val attentionMaskTensor = TensorData.LongBufferTensor(LongBuffer.wrap(attentionMask), inputShape)

        val output = model.run(
            mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor
            )
        )

        val embeddings = (output.values.first() as Array<FloatArray>)[0]
        normalizeL2(embeddings)
    }


    override suspend fun embedBatch(data: List<String>): List<FloatArray> {
        val allEmbeddings = mutableListOf<FloatArray>()
        val processor = object : BatchProcessor<String, FloatArray>(
            context.applicationContext as Application
        ) {
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
