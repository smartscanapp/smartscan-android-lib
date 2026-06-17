package com.fpf.smartscansdk.ml.ocr.preprocess

import android.graphics.Bitmap
import com.fpf.smartscansdk.core.media.resizeToMultipleOf32

internal data class DetPreprocessResult(
    val tensorData: FloatArray,
    val shape: LongArray,
    val originalH: Int,
    val originalW: Int,
)

internal object DetPreprocessor {

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)
    private const val scale = 1f / 255f

    fun preprocess(
        bitmap: Bitmap,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int,
        imgMode: String,
    ): DetPreprocessResult {

        val originalH = bitmap.height
        val originalW = bitmap.width

        val resized = resizeToMultipleOf32(
            bitmap,
            limitSideLen,
            limitType,
            maxSideLimit
        )

        val w = resized.width
        val h = resized.height

        val pixels = IntArray(w * h)
        resized.getPixels(pixels, 0, w, 0, 0, w, h)

        val tensor = FloatArray(3 * w * h)

        val isRGB = imgMode.uppercase() == "RGB"

        for (i in pixels.indices) {
            val c = pixels[i]

            var r = (c shr 16) and 0xFF
            var g = (c shr 8) and 0xFF
            var b = c and 0xFF

            // Bitmap is RGB already; OpenCV BGR expectation handled here
            if (!isRGB) {
                val tmp = r
                r = b
                b = tmp
            }

            val rf = r * scale
            val gf = g * scale
            val bf = b * scale

            tensor[i] = (rf - mean[0]) / std[0]
            tensor[i + w * h] = (gf - mean[1]) / std[1]
            tensor[i + 2 * w * h] = (bf - mean[2]) / std[2]
        }

        return DetPreprocessResult(
            tensorData = tensor,
            shape = longArrayOf(1, 3, h.toLong(), w.toLong()),
            originalH = originalH,
            originalW = originalW
        )
    }
}