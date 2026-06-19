package com.fpf.smartscansdk.ml.embeddings.minilm

import ai.onnxruntime.OnnxTensor
import android.content.Context
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.copyFloatBuffer
import com.fpf.smartscansdk.core.embeddings.TextEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceLoader
import kotlinx.coroutines.*
import java.nio.LongBuffer
import kotlin.collections.toLongArray

class MiniLMTextEmbedder(
    context: Context,
    modelSource: ModelAssetSource,
    vocabSource: ModelAssetSource,
    configSource: ModelAssetSource,
    override val maxTokens: Int = 128, // used as param because sentence transformer models can be exported with different token lengths
    ) : TextEmbeddingProvider {

    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileLoader(modelSource.file))
    }

    private val tokenizer: MiniLmTokenizer = when {
        vocabSource is ModelAssetSource.Resource && configSource is ModelAssetSource.Resource ->
            MiniLmTokenizer.load(context, vocabSource.resId, configSource.resId)

        vocabSource is ModelAssetSource.LocalFile && configSource is ModelAssetSource.LocalFile ->
            MiniLmTokenizer.load(vocabSource.file, configSource.file)

        else -> throw SmartScanException.InvalidTextEmbedderResourceFiles()
    }

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
        val inputIdsTensor = OnnxTensor.createTensor(
            model.getEnv(),
            LongBuffer.wrap(ids.map { it.toLong() }.toLongArray()),
            shape
        )
        val attentionMaskTensor = OnnxTensor.createTensor(
            model.getEnv(),
            LongBuffer.wrap(mask.map { it.toLong() }.toLongArray()),
            shape
        )
        val output = model.run(mapOf("input_ids" to inputIdsTensor, "attention_mask" to attentionMaskTensor))
        val embedding = output.values.first() as OnnxTensor
        try {
            normalizeL2(copyFloatBuffer((embedding).floatBuffer))
        }finally {
            output.values.forEach { it.close() }
        }
    }

    override fun closeSession()  = model.close()

}
