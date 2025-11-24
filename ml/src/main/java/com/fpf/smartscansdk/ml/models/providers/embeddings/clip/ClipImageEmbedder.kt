package com.fpf.smartscansdk.ml.models.providers.embeddings.clip

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.get
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.core.media.centerCrop
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.ml.data.FilePath
import com.fpf.smartscansdk.ml.data.ModelSource
import com.fpf.smartscansdk.ml.data.ResourceId
import com.fpf.smartscansdk.ml.data.TensorData
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.ResourceOnnxLoader
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

// Using ModelSource enables using with bundle model or local model which has been downloaded
class ClipImageEmbedder(private val context: Context, modelSource: ModelSource, ) : ImageEmbeddingProvider {
    companion object {
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 3
        const val IMAGE_SIZE_X = 224
        const val IMAGE_SIZE_Y = 224
        val MEAN = floatArrayOf(0.48145467f, 0.4578275f, 0.40821072f)
        val STD  = floatArrayOf(0.26862955f, 0.2613026f, 0.2757771f)
    }
    private val model: OnnxModel = when(modelSource){
        is FilePath -> OnnxModel(FileOnnxLoader(modelSource.path))
        is ResourceId -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
    }

    override val embeddingDim: Int = 512
    private var closed = false

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw IllegalStateException("Model not initialized")

        val inputShape = longArrayOf(DIM_BATCH_SIZE.toLong(), DIM_PIXEL_SIZE.toLong(), IMAGE_SIZE_Y.toLong(), IMAGE_SIZE_X.toLong())
        val imgData: FloatBuffer = preProcess(data)
        val inputName = model.getInputNames()?.firstOrNull() ?: throw IllegalStateException("Model inputs not available")
        val output = model.run(mapOf(inputName to TensorData.FloatBufferTensor(imgData, inputShape)))
        normalizeL2((output.values.first() as Array<FloatArray>)[0])
    }

    override suspend fun embedBatch(data: List<Bitmap>): List<FloatArray> {
        val allEmbeddings = mutableListOf<FloatArray>()

        val processor = object : BatchProcessor<Bitmap, FloatArray>(context = context.applicationContext as Application) {
            override suspend fun onProcess(context: Context, item: Bitmap): FloatArray {
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

    private fun preProcess(bitmap: Bitmap): FloatBuffer {
        val cropped = centerCrop(bitmap, IMAGE_SIZE_X)
        val numFloats = DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_Y * IMAGE_SIZE_X
        val byteBuffer = ByteBuffer.allocateDirect(numFloats * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
        for (c in 0 until DIM_PIXEL_SIZE) {
            for (y in 0 until IMAGE_SIZE_Y) {
                for (x in 0 until IMAGE_SIZE_X) {
                    val px = cropped[x, y]
                    val v = when (c) {
                        0 -> (px shr 16 and 0xFF) / 255f  // R
                        1 -> (px shr  8 and 0xFF) / 255f  // G
                        else -> (px and 0xFF) / 255f  // B
                    }
                    val norm = (v - MEAN[c]) / STD[c]
                    floatBuffer.put(norm)
                }
            }
        }
        floatBuffer.rewind()
        return floatBuffer
    }
}
