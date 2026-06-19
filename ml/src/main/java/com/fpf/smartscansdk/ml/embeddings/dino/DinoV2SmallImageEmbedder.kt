package com.fpf.smartscansdk.ml.embeddings.dino

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.get
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.copyFloatBuffer
import com.fpf.smartscansdk.core.embeddings.ImageEmbeddingProvider
import com.fpf.smartscansdk.core.embeddings.normalizeL2
import com.fpf.smartscansdk.core.media.centerCrop
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.loaders.FileLoader
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.ResourceLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class DinoV2SmallImageEmbedder(
    context: Context,
    modelSource: ModelAssetSource,
) : ImageEmbeddingProvider {

    companion object  {
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 3
        const val IMAGE_SIZE_X = 224
        const val IMAGE_SIZE_Y = 224
        val MEAN= floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD=floatArrayOf(0.229f, 0.224f, 0.225f)
    }
    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileLoader(modelSource.file))
    }

    override val embeddingDim: Int = 384

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    override suspend fun embed(data: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        if (!isInitialized()) throw SmartScanException.ModelNotInitialised()

        val inputShape = longArrayOf(DIM_BATCH_SIZE.toLong(), DIM_PIXEL_SIZE.toLong(), IMAGE_SIZE_Y.toLong(), IMAGE_SIZE_X.toLong())
        val imgData: FloatBuffer = preProcess(data)
        val inputName = model.getInputNames()!!.first()
        val output = model.run(mapOf(inputName to OnnxTensor.createTensor(model.getEnv(), imgData, inputShape)))
        val embedding = output.values.first() as OnnxTensor
        try {
            normalizeL2(copyFloatBuffer((embedding).floatBuffer))
        }finally {
            output.values.forEach { it.close() }
        }
    }

    override fun closeSession()  = model.close()

    private fun preProcess(bitmap: Bitmap): FloatBuffer {
        val cropped = centerCrop(bitmap, IMAGE_SIZE_X)
        val numFloats = DIM_BATCH_SIZE * DIM_PIXEL_SIZE * IMAGE_SIZE_X * IMAGE_SIZE_Y
        val byteBuffer = ByteBuffer.allocateDirect(numFloats * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()

        for (c in 0 until DIM_PIXEL_SIZE) { // R, G, B channels
            for (y in 0 until IMAGE_SIZE_X) {
                for (x in 0 until IMAGE_SIZE_X) {
                    val px = cropped[x, y]
                    val v = when (c) {
                        0 -> ((px shr 16) and 0xFF) / 255f // R
                        1 -> ((px shr 8) and 0xFF) / 255f  // G
                        else -> (px and 0xFF) / 255f       // B
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