package com.fpf.smartscansdk.ml.providers.embeddings.minilm

import android.app.Application
import android.content.Context
import com.fpf.smartscansdk.core.embeddings.TextEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceOnnxLoader
import com.fpf.smartscansdk.ml.data.FilePath
import com.fpf.smartscansdk.ml.data.ModelSource
import com.fpf.smartscansdk.ml.data.ResourceId
import com.fpf.smartscansdk.ml.data.TensorData
import kotlinx.coroutines.*
import java.nio.LongBuffer
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.ml.R

class MiniLMTextEmbedder(
    private val context: Context,
    modelSource: ModelSource,
) : TextEmbeddingProvider {
    private val model: OnnxModel = when (modelSource) {
        is FilePath -> OnnxModel(FileOnnxLoader(modelSource.path))
        is ResourceId -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
    }

    private var tokenizer = MiniLmTokenizer.fromRawResources(context, R.raw.minilm_vocab,  R.raw.minilm_tokenizer_config)
    private var closed = false
    override val embeddingDim: Int = 384 // MiniLM-L6-v2 dimension

    override suspend fun initialize()  {
        model.loadModel()
    }

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: String): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw IllegalStateException("Model not initialized")

        val maxLen = 128  // must match ONNX export
        val (rawIds, rawMask) = tokenizer.encode(data)
        val inputIds = rawIds.copyOfRange(0, maxLen)
        val attentionMask = rawMask.copyOfRange(0, maxLen)
        val inputShape = longArrayOf(1, maxLen.toLong())
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
