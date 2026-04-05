package com.fpf.smartscansdk.ml.providers.embeddings.minilm

import android.app.Application
import android.content.Context
import com.fpf.smartscansdk.core.embeddings.TextEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.core.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceOnnxLoader
import com.fpf.smartscansdk.ml.models.TensorData
import kotlinx.coroutines.*
import java.nio.LongBuffer
import com.fpf.smartscansdk.core.processors.BatchProcessor

class MiniLMTextEmbedder(
    private val context: Context,
    modelSource: ModelAssetSource,
    vocabSource: ModelAssetSource,
    configSource: ModelAssetSource,
    override val maxTokens: Int = 128, // used as param because sentence transformer models can be exported with different token lengths
    ) : TextEmbeddingProvider {

    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileOnnxLoader(modelSource.file))
    }

    private val tokenizer: MiniLmTokenizer = when {
        vocabSource is ModelAssetSource.Resource && configSource is ModelAssetSource.Resource ->
            MiniLmTokenizer.load(context, vocabSource.resId, configSource.resId)

        vocabSource is ModelAssetSource.LocalFile && configSource is ModelAssetSource.LocalFile ->
            MiniLmTokenizer.load(vocabSource.file, configSource.file)

        else -> error("vocabSource and configSource must be of the same type")
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
