package com.fpf.smartscansdk.ml.providers.ocr.util

import android.graphics.Bitmap
import kotlin.math.max

object ImageUtils {

    fun resizeToMultipleOf32(
        src: Bitmap,
        limitSideLen: Int,
        limitType: String,
        maxSideLimit: Int,
    ): Bitmap {

        val w = src.width
        val h = src.height

        var ratio = when (limitType.lowercase()) {
            "max" -> if (maxOf(h, w) > limitSideLen)
                limitSideLen.toDouble() / maxOf(h, w)
            else 1.0

            "min" -> if (minOf(h, w) < limitSideLen)
                limitSideLen.toDouble() / minOf(h, w)
            else 1.0

            "resize_long" ->
                limitSideLen.toDouble() / maxOf(h, w)

            else -> throw IllegalArgumentException("Unsupported det limit type: $limitType")
        }

        var newH = (h * ratio).toInt()
        var newW = (w * ratio).toInt()

        if (maxOf(newH, newW) > maxSideLimit) {
            ratio = maxSideLimit.toDouble() / maxOf(newH, newW)
            newH = (newH * ratio).toInt()
            newW = (newW * ratio).toInt()
        }

        newH = max(roundHalfToEven(newH / 32.0) * 32, 32)
        newW = max(roundHalfToEven(newW / 32.0) * 32, 32)

        return Bitmap.createScaledBitmap(src, newW, newH, true)
    }

    private fun roundHalfToEven(value: Double): Int {
        val floor = value.toInt()
        val diff = value - floor

        return when {
            diff > 0.5 -> floor + 1
            diff < 0.5 -> floor
            else -> if (floor % 2 == 0) floor else floor + 1
        }
    }
}