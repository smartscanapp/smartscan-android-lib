package com.fpf.smartscansdk.ml.embeddings.clip

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

class ClipTextEmbedder(
    context: Context,
    modelSource: ModelAssetSource,
    vocabSource: ModelAssetSource,
    mergesSource: ModelAssetSource,
    ) : TextEmbeddingProvider {

    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileLoader(modelSource.file))
    }

    private val tokenizer: ClipTokenizer = when {
        vocabSource is ModelAssetSource.Resource && mergesSource is ModelAssetSource.Resource ->
            ClipTokenizer.load(context, vocabSource.resId, mergesSource.resId)

        vocabSource is ModelAssetSource.LocalFile && mergesSource is ModelAssetSource.LocalFile ->
            ClipTokenizer.load(vocabSource.file, mergesSource.file)

        else -> throw SmartScanException.InvalidTextEmbedderResourceFiles()
    }
    private val tokenBOS = 49406
    private val tokenEOS = 49407

    override val embeddingDim: Int = 512
    override val maxTokens: Int = 77

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw SmartScanException.ModelNotInitialised()

        val clean = Regex("[^A-Za-z0-9 ]").replace(data, "").lowercase()
        val tokens = (mutableListOf(tokenBOS) + tokenizer.encode(clean) + tokenEOS).take(maxTokens).toMutableList()
        if (tokens.size < maxTokens) tokens += List(maxTokens - tokens.size) { 0 }

        val inputIds = LongBuffer.allocate(1 * maxTokens).apply {
            tokens.forEach { put(it.toLong()) }
            rewind()
        }
        val inputShape = longArrayOf(1, maxTokens.toLong())
        val inputName = model.getInputNames()!!.first()
        val output = model.run(mapOf(inputName to OnnxTensor.createTensor(model.getEnv(), inputIds, inputShape)))
        val embedding = output.values.first() as OnnxTensor
        try {
            normalizeL2(copyFloatBuffer((embedding).floatBuffer))
        }finally {
            output.values.forEach { it.close() }
        }
    }

    override fun closeSession()  = model.close()
}
