package com.fpf.smartscansdk.ml.providers.ocr.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory

internal object BitmapUtils {

    fun imdecodeBGR(imageBytes: ByteArray): Bitmap {
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
        }

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
            ?: throw IllegalArgumentException("Failed to decode image bytes")
    }

    fun bitmapToBGRBitmap(bitmap: Bitmap): Bitmap {
        return swapRedBlue(bitmap)
    }

    fun bitmapToRGBBitmap(bitmap: Bitmap): Bitmap {
        return ensureArgb8888(bitmap)
    }

    fun bgrBitmapToBitmap(bitmap: Bitmap): Bitmap {
        return swapRedBlue(bitmap)
    }

    private fun ensureArgb8888(src: Bitmap): Bitmap {
        return if (src.config == Bitmap.Config.ARGB_8888) {
            if (src.isMutable) src else src.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            src.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    private fun swapRedBlue(src: Bitmap): Bitmap {
        val input = ensureArgb8888(src)
        val width = input.width
        val height = input.height

        val pixels = IntArray(width * height)
        input.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            val c = pixels[i]
            val a = c and -0x1000000
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            pixels[i] = a or (b shl 16) or (g shl 8) or r
        }

        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }
}