package com.fpf.smartscansdk.ml.providers.embeddings.minilm

import android.content.Context
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.embeddings.TextEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceOnnxLoader
import com.fpf.smartscansdk.ml.models.TensorData
import kotlinx.coroutines.*
import java.nio.LongBuffer
import kotlin.collections.toLongArray

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

        else -> throw SmartScanException.InvalidTextEmbedderResourceFiles()
    }

    private var closed = false
    override val embeddingDim: Int = 384

    override suspend fun initialize()  {
        model.loadModel()
    }

    override fun isInitialized() = model.isLoaded()


    override suspend fun embed(data: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw SmartScanException.ModelNotInitialised()

        var (ids, mask) = tokenizer.encode(data)

        if (ids.size > maxTokens) {
            ids = ids.copyOf(maxTokens)
            mask = mask.copyOf(maxTokens)
        } else if (ids.size < maxTokens) {
            val paddedIds = IntArray(maxTokens) // padtoken is 0
            val paddedMask = IntArray(maxTokens)

            System.arraycopy(ids, 0, paddedIds, 0, ids.size)
            System.arraycopy(mask, 0, paddedMask, 0, mask.size)

            ids = paddedIds
            mask = paddedMask
        }

        val shape = longArrayOf(1, maxTokens.toLong())
        val inputIdsTensor = TensorData.LongBufferTensor(
            LongBuffer.wrap(ids.map { it.toLong() }.toLongArray()),
            shape
        )
        val attentionMaskTensor = TensorData.LongBufferTensor(
            LongBuffer.wrap(mask.map { it.toLong() }.toLongArray()),
            shape
        )

        val output = model.run(
            mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor
            )
        )

        val embeddings = (output.values.first() as Array<FloatArray>)[0]
        normalizeL2(embeddings)
    }

    override fun closeSession() {
        if (closed) return
        closed = true
        (model as? AutoCloseable)?.close()
    }
}
