package com.fpf.smartscansdk.ml.ocr.postprocess

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import com.fpf.smartscansdk.ml.ocr.model.OCRBox
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

internal object QuadTextCrop {
    private const val VERTICAL_CROP_RATIO = 1.5

    fun crop(src: Bitmap, box: OCRBox): Bitmap {
        val source = ensureArgb8888(src)

        val ordered = QuadGeometry.orderMinAreaRectPoints(box.points.toTypedArray())

        val widthTop = hypot(
            ordered[0].x - ordered[1].x,
            ordered[0].y - ordered[1].y,
        )
        val widthBottom = hypot(
            ordered[2].x - ordered[3].x,
            ordered[2].y - ordered[3].y,
        )
        val heightLeft = hypot(
            ordered[0].x - ordered[3].x,
            ordered[0].y - ordered[3].y,
        )
        val heightRight = hypot(
            ordered[1].x - ordered[2].x,
            ordered[1].y - ordered[2].y,
        )

        val dstW = max(widthTop, widthBottom).roundToInt().coerceAtLeast(1)
        val dstH = max(heightLeft, heightRight).roundToInt().coerceAtLeast(1)

        val dstCorners = listOf(
            PointF(0f, 0f),
            PointF(dstW.toFloat(), 0f),
            PointF(dstW.toFloat(), dstH.toFloat()),
            PointF(0f, dstH.toFloat()),
        )

        val h = computeHomography(dstCorners, ordered)
            ?: return Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)

        val srcPixels = IntArray(source.width * source.height)
        source.getPixels(srcPixels, 0, source.width, 0, 0, source.width, source.height)

        val outPixels = IntArray(dstW * dstH)

        for (y in 0 until dstH) {
            for (x in 0 until dstW) {
                val p = mapPoint(h, x + 0.5, y + 0.5)
                outPixels[y * dstW + x] = sampleBilinear(
                    pixels = srcPixels,
                    srcW = source.width,
                    srcH = source.height,
                    x = p.first,
                    y = p.second,
                )
            }
        }

        val cropped = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        cropped.setPixels(outPixels, 0, dstW, 0, 0, dstW, dstH)

        if (cropped.height.toDouble() / cropped.width.toDouble() >= VERTICAL_CROP_RATIO) {
            val matrix = Matrix().apply { postRotate(-90f) }
            val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
            cropped.recycle()
            return rotated
        }

        return cropped
    }

    private fun ensureArgb8888(src: Bitmap): Bitmap {
        return if (src.config == Bitmap.Config.ARGB_8888) {
            if (src.isMutable) src else src.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            src.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    private fun computeHomography(src: List<PointF>, dst: List<PointF>): DoubleArray? {
        if (src.size != 4 || dst.size != 4) return null

        val a = Array(8) { DoubleArray(9) }

        for (i in 0 until 4) {
            val x = src[i].x.toDouble()
            val y = src[i].y.toDouble()
            val u = dst[i].x.toDouble()
            val v = dst[i].y.toDouble()

            val r = i * 2

            a[r][0] = x
            a[r][1] = y
            a[r][2] = 1.0
            a[r][3] = 0.0
            a[r][4] = 0.0
            a[r][5] = 0.0
            a[r][6] = -u * x
            a[r][7] = -u * y
            a[r][8] = u

            a[r + 1][0] = 0.0
            a[r + 1][1] = 0.0
            a[r + 1][2] = 0.0
            a[r + 1][3] = x
            a[r + 1][4] = y
            a[r + 1][5] = 1.0
            a[r + 1][6] = -v * x
            a[r + 1][7] = -v * y
            a[r + 1][8] = v
        }

        for (col in 0 until 8) {
            var pivot = col
            for (row in col + 1 until 8) {
                if (abs(a[row][col]) > abs(a[pivot][col])) pivot = row
            }
            if (abs(a[pivot][col]) < 1e-10) return null
            if (pivot != col) {
                val tmp = a[pivot]
                a[pivot] = a[col]
                a[col] = tmp
            }

            val div = a[col][col]
            for (j in col until 9) a[col][j] /= div

            for (row in 0 until 8) {
                if (row == col) continue
                val factor = a[row][col]
                if (factor == 0.0) continue
                for (j in col until 9) {
                    a[row][j] -= factor * a[col][j]
                }
            }
        }

        return doubleArrayOf(
            a[0][8], a[1][8], a[2][8],
            a[3][8], a[4][8], a[5][8],
            a[6][8], a[7][8], 1.0
        )
    }

    private fun mapPoint(h: DoubleArray, x: Double, y: Double): Pair<Double, Double> {
        val denom = h[6] * x + h[7] * y + h[8]
        if (abs(denom) < 1e-10) return 0.0 to 0.0

        val sx = (h[0] * x + h[1] * y + h[2]) / denom
        val sy = (h[3] * x + h[4] * y + h[5]) / denom
        return sx to sy
    }

    private fun sampleBilinear(
        pixels: IntArray,
        srcW: Int,
        srcH: Int,
        x: Double,
        y: Double,
    ): Int {
        val clampedX = x.coerceIn(0.0, (srcW - 1).toDouble())
        val clampedY = y.coerceIn(0.0, (srcH - 1).toDouble())

        val x0 = clampedX.toInt()
        val y0 = clampedY.toInt()
        val x1 = minOf(x0 + 1, srcW - 1)
        val y1 = minOf(y0 + 1, srcH - 1)

        val dx = clampedX - x0
        val dy = clampedY - y0

        val c00 = pixels[y0 * srcW + x0]
        val c10 = pixels[y0 * srcW + x1]
        val c01 = pixels[y1 * srcW + x0]
        val c11 = pixels[y1 * srcW + x1]

        val a = bilerp(
            (c00 ushr 24) and 0xFF,
            (c10 ushr 24) and 0xFF,
            (c01 ushr 24) and 0xFF,
            (c11 ushr 24) and 0xFF,
            dx, dy
        )
        val r = bilerp(
            (c00 shr 16) and 0xFF,
            (c10 shr 16) and 0xFF,
            (c01 shr 16) and 0xFF,
            (c11 shr 16) and 0xFF,
            dx, dy
        )
        val g = bilerp(
            (c00 shr 8) and 0xFF,
            (c10 shr 8) and 0xFF,
            (c01 shr 8) and 0xFF,
            (c11 shr 8) and 0xFF,
            dx, dy
        )
        val b = bilerp(
            c00 and 0xFF,
            c10 and 0xFF,
            c01 and 0xFF,
            c11 and 0xFF,
            dx, dy
        )

        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }

    private fun bilerp(c00: Int, c10: Int, c01: Int, c11: Int, dx: Double, dy: Double): Int {
        val top = c00 + (c10 - c00) * dx
        val bottom = c01 + (c11 - c01) * dx
        return (top + (bottom - top) * dy).roundToInt().coerceIn(0, 255)
    }
}