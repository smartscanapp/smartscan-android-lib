package com.fpf.smartscansdk.ml.ocr.preprocess

import android.graphics.Bitmap
import kotlin.math.ceil
import androidx.core.graphics.scale

internal data class RecPreprocessResult(
    val tensorData: FloatArray,
    val shape: LongArray,
)

internal object RecPreprocessor {

    private const val FIXED_HEIGHT = 48
    private const val MAX_IMG_W = 3200

    fun preprocessBatch(bitmaps: List<Bitmap>): RecPreprocessResult {

        val resized = ArrayList<Bitmap>(bitmaps.size)

        for (bmp in bitmaps) {
            val w = bmp.width
            val h = bmp.height
            val aspectRatio = if (h > 0) w.toDouble() / h else 1.0

            val newW = ceil(FIXED_HEIGHT * aspectRatio)
                .toInt()
                .coerceAtMost(MAX_IMG_W)

            resized.add(
                bmp.scale(newW, FIXED_HEIGHT)
            )
        }

        val n = resized.size
        val maxW = resized.maxOf { it.width }

        val channelSize = FIXED_HEIGHT * maxW
        val tensor = FloatArray(n * 3 * channelSize)

        for (b in resized.indices) {
            val bmp = resized[b]

            val w = bmp.width
            val h = bmp.height

            val pixels = IntArray(w * h)
            bmp.getPixels(pixels, 0, w, 0, 0, w, h)

            val base = b * 3 * channelSize

            for (i in pixels.indices) {
                val c = pixels[i]

                var r = (c shr 16) and 0xFF
                var g = (c shr 8) and 0xFF
                var bch = c and 0xFF

                val rf = (r / 255f - 0.5f) / 0.5f
                val gf = (g / 255f - 0.5f) / 0.5f
                val bf = (bch / 255f - 0.5f) / 0.5f

                val idx = i

                tensor[base + idx] = rf
                tensor[base + channelSize + idx] = gf
                tensor[base + 2 * channelSize + idx] = bf
            }

            bmp.recycle()
        }

        return RecPreprocessResult(
            tensorData = tensor,
            shape = longArrayOf(n.toLong(), 3, FIXED_HEIGHT.toLong(), maxW.toLong())
        )
    }
}