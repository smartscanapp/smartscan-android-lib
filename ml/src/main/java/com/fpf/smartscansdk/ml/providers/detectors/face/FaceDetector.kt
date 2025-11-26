package com.fpf.smartscansdk.ml.providers.detectors.face

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.scale
import com.fpf.smartscansdk.core.embeddings.IDetectorProvider
import com.fpf.smartscansdk.core.media.nms
import com.fpf.smartscansdk.ml.data.FilePath
import com.fpf.smartscansdk.ml.data.ModelSource
import com.fpf.smartscansdk.ml.data.ResourceId
import com.fpf.smartscansdk.ml.data.TensorData
import com.fpf.smartscansdk.ml.models.FileOnnxLoader
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.ResourceOnnxLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer


class FaceDetector(
    context: Context,
    modelSource: ModelSource,
    private val confThreshold: Float = 0.5f,
    private val nmsThreshold: Float = 0.3f
) : IDetectorProvider<Bitmap> {
    private val model: OnnxModel = when(modelSource){
        is FilePath -> OnnxModel(FileOnnxLoader(modelSource.path))
        is ResourceId -> OnnxModel(ResourceOnnxLoader(context.resources, modelSource.resId))
    }

    companion object {
        private const val TAG = "FaceDetector"
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 3
        const val IMAGE_SIZE_X = 320
        const val IMAGE_SIZE_Y = 240
    }

    override suspend fun initialize() = model.loadModel()

    override fun isInitialized() = model.isLoaded()

    private var closed = false

    override suspend fun detect(data: Bitmap): Pair<List<Float>, List<FloatArray>> = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()
        val inputShape = longArrayOf(DIM_BATCH_SIZE.toLong(), DIM_PIXEL_SIZE.toLong(), IMAGE_SIZE_Y.toLong(), IMAGE_SIZE_X.toLong())
        val imgData: FloatBuffer = preProcess(data)
        val inputName = model.getInputNames()?.firstOrNull() ?: throw IllegalStateException("Model inputs not available")
        val outputs = model.run(mapOf(inputName to TensorData.FloatBufferTensor(imgData, inputShape)))

        val outputList = outputs.values.toList()
        @Suppress("UNCHECKED_CAST")
        val scoresRawFull  = outputList[0] as Array<Array<FloatArray>>
        @Suppress("UNCHECKED_CAST")
        val boxesRawFull = outputList[1] as Array<Array<FloatArray>>

        // Extract the first element (batch dimension)
        val scoresRaw = scoresRawFull[0]  // shape: [num_boxes, 2]
        val boxesRaw = boxesRawFull[0]    // shape: [num_boxes, 4]

        val imgWidth = data.width
        val imgHeight = data.height

        val boxesList = mutableListOf<FloatArray>()
        val scoresList = mutableListOf<Float>()
        for (i in scoresRaw.indices) {
            val faceScore = scoresRaw[i][1]
            if (faceScore > confThreshold) {
                val box = boxesRaw[i]
                // Box values are normalized; convert to absolute pixel coordinates.
                val x1 = box[0] * imgWidth
                val y1 = box[1] * imgHeight
                val x2 = box[2] * imgWidth
                val y2 = box[3] * imgHeight
                boxesList.add(floatArrayOf(x1, y1, x2, y2))
                scoresList.add(faceScore)
            }
        }

        val inferenceTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "Detection Inference Time: $inferenceTime ms")

        // Apply NMS if any detection exists.
        if (boxesList.isNotEmpty()) {
            val keepIndices = nms(boxesList, scoresList, nmsThreshold)
            val filteredBoxes = keepIndices.map { boxesList[it] }
            val filteredScores = keepIndices.map { scoresList[it] }
            return@withContext Pair(filteredScores, filteredBoxes)
        } else {
            return@withContext Pair(emptyList<Float>(), emptyList<FloatArray>())
        }
    }

    private fun preProcess(bitmap: Bitmap): FloatBuffer {
        val resizedBitmap = bitmap.scale(IMAGE_SIZE_X, IMAGE_SIZE_Y)
        val width = resizedBitmap.width
        val height = resizedBitmap.height
        val intValues = IntArray(width * height)
        resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        val floatArray = FloatArray(DIM_PIXEL_SIZE * height * width)

        // Process each pixel and store them in channel-first order.
        // Channel 0: indices 0 .. height*width-1, etc.
        for (i in 0 until height) {
            for (j in 0 until width) {
                val pixel = intValues[i * width + j]
                val r = ((pixel shr 16) and 0xFF).toFloat()
                val g = ((pixel shr 8) and 0xFF).toFloat()
                val b = (pixel and 0xFF).toFloat()

                // Normalize channels
                val normalizedR = (r - 127f) / 128f
                val normalizedG = (g - 127f) / 128f
                val normalizedB = (b - 127f) / 128f

                val index = i * width + j
                floatArray[index] = normalizedR
                floatArray[height * width + index] = normalizedG
                floatArray[2 * height * width + index] = normalizedB
            }
        }

        val byteBuffer = ByteBuffer.allocateDirect(floatArray.size * 4).order(ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
        floatBuffer.put(floatArray)
        floatBuffer.position(0)
        return floatBuffer
    }

    override fun closeSession() {
        if (closed) return
        closed = true
        (model as? AutoCloseable)?.close()
    }
}
