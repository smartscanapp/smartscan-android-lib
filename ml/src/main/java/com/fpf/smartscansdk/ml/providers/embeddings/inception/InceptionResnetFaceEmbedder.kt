package com.fpf.smartscansdk.ml.providers.embeddings.inception

import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.media.centerCrop
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.TensorData
import com.fpf.smartscansdk.ml.models.loaders.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.ResourceOnnxLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

class InceptionResnetFaceEmbedder(
    private val context: Context,
    modelSource: ModelAssetSource,
) : ImageEmbeddingProvider {
    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileOnnxLoader(modelSource.file))
    }

    companion object {
        private const val TAG = "InceptionResnetFaceEmbedder"
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
        if (!isInitialized()) throw SmartScanException.ModelNotInitialised()

        val imgData = preProcess(data)
        val inputShape = longArrayOf(DIM_BATCH_SIZE.toLong(), DIM_PIXEL_SIZE.toLong(), IMAGE_SIZE_Y.toLong(), IMAGE_SIZE_X.toLong())
        val inputName = model.getInputNames()!!.first()
        val output = model.run(mapOf(inputName to TensorData.FloatBufferTensor(imgData, inputShape)))
        (output.values.first() as Array<FloatArray>)[0]
    }

    private fun preProcess(bitmap: Bitmap): FloatBuffer {
        val centredBitmap = centerCrop(bitmap, IMAGE_SIZE_X)
        val imgData = FloatBuffer.allocate(DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_X * IMAGE_SIZE_Y)
        imgData.rewind()
        val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
        val bmpData = IntArray(stride)
        centredBitmap.getPixels(bmpData, 0, centredBitmap.width, 0, 0, centredBitmap.width, centredBitmap.height)
        for (i in 0..<IMAGE_SIZE_X) {
            for (j in 0..<IMAGE_SIZE_Y) {
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
