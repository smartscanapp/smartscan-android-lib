package com.fpf.smartscansdk.ml.models.providers.embeddings.inception

import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.media.centerCrop
import com.fpf.smartscansdk.core.processors.BatchProcessor
import com.fpf.smartscansdk.ml.data.FilePath
import com.fpf.smartscansdk.ml.data.ModelSource
import com.fpf.smartscansdk.ml.data.ResourceId
import com.fpf.smartscansdk.ml.data.TensorData
import com.fpf.smartscansdk.ml.models.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.ResourceOnnxLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

class InceptionResnetFaceEmbedder(
    private val context: Context,
    modelSource: ModelSource,
) : ImageEmbeddingProvider {
    private val model: OnnxModel = when(modelSource){
        is FilePath -> OnnxModel(FileOnnxLoader(modelSource.path))
        is ResourceId -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
    }

    companion object {
        private const val TAG = "FaceEmbedder"
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 3
        const val IMAGE_SIZE_X = 160
        const val IMAGE_SIZE_Y = 160
        val MEAN= floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD=floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    override val embeddingDim: Int = 512
    private var closed = false

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw IllegalStateException("Model not initialized")

        val imgData = preProcess(data)
        val inputShape = longArrayOf(DIM_BATCH_SIZE.toLong(), DIM_PIXEL_SIZE.toLong(), IMAGE_SIZE_X.toLong(), IMAGE_SIZE_Y.toLong())
        val inputName = model.getInputNames()?.firstOrNull() ?: throw IllegalStateException("Model inputs not available")
        val output = model.run(mapOf(inputName to TensorData.FloatBufferTensor(imgData, inputShape)))
        (output.values.first() as Array<FloatArray>)[0]
    }

    override suspend fun embedBatch(data: List<Bitmap>): List<FloatArray> {
        val allEmbeddings = mutableListOf<FloatArray>()

        val processor = object : BatchProcessor<Bitmap, FloatArray>(context = context.applicationContext) {
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

    private fun preProcess(bitmap: Bitmap): FloatBuffer {
        val centredBitmap = centerCrop(bitmap, IMAGE_SIZE_X)
        val imgData = FloatBuffer.allocate(DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_X * IMAGE_SIZE_Y)
        imgData.rewind()
        val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
        val bmpData = IntArray(stride)
        centredBitmap.getPixels(bmpData, 0, centredBitmap.width, 0, 0, centredBitmap.width, centredBitmap.height)
        for (i in 0..IMAGE_SIZE_X - 1) {
            for (j in 0..IMAGE_SIZE_Y - 1) {
                val idx = IMAGE_SIZE_Y * i + j
                val pixelValue = bmpData[idx]
                imgData.put(idx, (((pixelValue shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]))
                imgData.put(idx + stride, (((pixelValue shr 8 and 0xFF) / 255f - MEAN[1]) / STD[1]))
                imgData.put(idx + stride * 2, (((pixelValue and 0xFF) / 255f - MEAN[2]) / STD[2]))
            }
        }

        imgData.rewind()
        return imgData
    }

    override fun closeSession() {
        if (closed) return
        closed = true
        (model as? AutoCloseable)?.close()
    }
}
