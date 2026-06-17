package com.fpf.smartscansdk.ml.ocr.postprocess

import android.graphics.PointF
import com.fpf.smartscansdk.ml.ocr.model.OCRBox
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

internal object DBPostProcessor {
    private const val MIN_SIZE_BEFORE_UNCLIP = 3f
    private const val MIN_SIZE_AFTER_UNCLIP = 5f

    private data class IPoint(val x: Int, val y: Int)

    fun process(
        pred: FloatArray,
        predShape: LongArray,
        thresh: Float,
        boxThresh: Float,
        unclipRatio: Float,
        maxCandidates: Int,
        useDilation: Boolean,
        scoreMode: String,
        boxType: String,
        originalH: Int,
        originalW: Int,
    ): List<OCRBox> {
        require(boxType == "quad") { "Only DBPostProcess box_type=quad is supported" }

        val pH = predShape[2].toInt()
        val pW = predShape[3].toInt()
        val scaleX = originalW.toDouble() / pW
        val scaleY = originalH.toDouble() / pH

        val normalizedScoreMode = scoreMode.lowercase()

        val mask = BooleanArray(pH * pW) { idx -> pred[idx] >= thresh }
        val workingMask = if (useDilation) dilate2x2(mask, pW, pH) else mask

        val components = findConnectedComponents(workingMask, pW, pH)
            .sortedByDescending { it.size }

        val boxes = mutableListOf<OCRBox>()

        for (component in components.take(maxCandidates)) {
            if (component.size < 4) continue

            val score = if (normalizedScoreMode == "slow") {
                meanScore(pred, pW, component)
            } else {
                meanScore(pred, pW, component)
            }
            if (score < boxThresh) continue

            val rect = pcaRotatedRect(component)
            val minSide = min(rect.width, rect.height)
            if (minSide < MIN_SIZE_BEFORE_UNCLIP) continue

            val expanded = expandRotatedRect(rect, unclipRatio)
            if (min(expanded.width, expanded.height) < MIN_SIZE_AFTER_UNCLIP) continue

            val points = orderedQuadPoints(expanded)
            val scaled = points.map { p ->
                scalePoint(p, scaleX, scaleY, originalW, originalH)
            }

            val boxW = hypot(
                scaled[1].x - scaled[0].x,
                scaled[1].y - scaled[0].y,
            )
            val boxH = hypot(
                scaled[3].x - scaled[0].x,
                scaled[3].y - scaled[0].y,
            )
            if (boxW <= 3 || boxH <= 3) continue

            boxes.add(OCRBox(points = scaled))
        }

        return boxes
    }

    private data class RotatedRect(
        val cx: Double,
        val cy: Double,
        val width: Double,
        val height: Double,
        val angleRad: Double,
    )

    private fun pcaRotatedRect(points: List<IPoint>): RotatedRect {
        var sumX = 0.0
        var sumY = 0.0
        for (p in points) {
            sumX += p.x
            sumY += p.y
        }
        val cx = sumX / points.size
        val cy = sumY / points.size

        var sxx = 0.0
        var sxy = 0.0
        var syy = 0.0
        for (p in points) {
            val dx = p.x - cx
            val dy = p.y - cy
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        }

        val angle = 0.5 * atan2(2.0 * sxy, sxx - syy)
        val cosA = cos(angle)
        val sinA = sin(angle)

        var minU = Double.POSITIVE_INFINITY
        var maxU = Double.NEGATIVE_INFINITY
        var minV = Double.POSITIVE_INFINITY
        var maxV = Double.NEGATIVE_INFINITY

        for (p in points) {
            val dx = p.x - cx
            val dy = p.y - cy
            val u = dx * cosA + dy * sinA
            val v = -dx * sinA + dy * cosA
            if (u < minU) minU = u
            if (u > maxU) maxU = u
            if (v < minV) minV = v
            if (v > maxV) maxV = v
        }

        val width = maxU - minU
        val height = maxV - minV
        return RotatedRect(cx, cy, width, height, angle)
    }

    private fun expandRotatedRect(rect: RotatedRect, ratio: Float): RotatedRect {
        val area = rect.width * rect.height
        val perimeter = 2.0 * (rect.width + rect.height)
        if (area <= 0.0 || perimeter <= 0.0) return rect

        val distance = area * ratio / perimeter
        val newW = rect.width + 2.0 * distance
        val newH = rect.height + 2.0 * distance

        return rect.copy(width = newW, height = newH)
    }

    private fun orderedQuadPoints(rect: RotatedRect): List<PointF> {
        val halfW = rect.width / 2.0
        val halfH = rect.height / 2.0
        val cosA = cos(rect.angleRad)
        val sinA = sin(rect.angleRad)

        fun map(localX: Double, localY: Double): PointF {
            val x = rect.cx + localX * cosA - localY * sinA
            val y = rect.cy + localX * sinA + localY * cosA
            return PointF(x.toFloat(), y.toFloat())
        }

        val pts = listOf(
            map(-halfW, -halfH),
            map(halfW, -halfH),
            map(halfW, halfH),
            map(-halfW, halfH),
        )
        return orderQuadPoints(pts)
    }

    private fun orderQuadPoints(points: List<PointF>): List<PointF> {
        val tl = points.minBy { it.x + it.y }
        val br = points.maxBy { it.x + it.y }
        val tr = points.minBy { it.x - it.y }
        val bl = points.maxBy { it.x - it.y }
        return listOf(tl, tr, br, bl)
    }

    private fun scalePoint(
        point: PointF,
        scaleX: Double,
        scaleY: Double,
        originalW: Int,
        originalH: Int,
    ): PointF {
        return PointF(
            roundHalfToEven(point.x * scaleX).coerceIn(0, originalW).toFloat(),
            roundHalfToEven(point.y * scaleY).coerceIn(0, originalH).toFloat(),
        )
    }

    private fun meanScore(pred: FloatArray, width: Int, points: List<IPoint>): Float {
        if (points.isEmpty()) return 0f
        var sum = 0.0
        for (p in points) {
            val idx = p.y * width + p.x
            if (idx in pred.indices) {
                sum += pred[idx].toDouble()
            }
        }
        return (sum / points.size).toFloat()
    }

    private fun dilate2x2(mask: BooleanArray, width: Int, height: Int): BooleanArray {
        val out = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var on = false
                for (dy in 0..1) {
                    val ny = y - dy
                    if (ny !in 0 until height) continue
                    for (dx in 0..1) {
                        val nx = x - dx
                        if (nx !in 0 until width) continue
                        if (mask[ny * width + nx]) {
                            on = true
                            break
                        }
                    }
                    if (on) break
                }
                out[y * width + x] = on
            }
        }
        return out
    }

    private fun findConnectedComponents(mask: BooleanArray, width: Int, height: Int): List<List<IPoint>> {
        val visited = BooleanArray(mask.size)
        val components = mutableListOf<List<IPoint>>()
        val queue = ArrayDeque<Int>()

        fun neighbors(idx: Int): IntArray {
            val x = idx % width
            val y = idx / width
            val result = IntArray(8)
            var n = 0
            for (dy in -1..1) {
                for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in 0 until width && ny in 0 until height) {
                        result[n++] = ny * width + nx
                    }
                }
            }
            return result.copyOf(n)
        }

        for (i in mask.indices) {
            if (!mask[i] || visited[i]) continue

            val component = mutableListOf<IPoint>()
            visited[i] = true
            queue.addLast(i)

            while (queue.isNotEmpty()) {
                val cur = queue.removeFirst()
                val x = cur % width
                val y = cur / width
                component.add(IPoint(x, y))

                for (nb in neighbors(cur)) {
                    if (!visited[nb] && mask[nb]) {
                        visited[nb] = true
                        queue.addLast(nb)
                    }
                }
            }

            components.add(component)
        }

        return components
    }

    fun roundHalfToEven(value: Double): Int {
        val floored = floor(value)
        val diff = value - floored
        return when {
            diff < 0.5 -> floored.toInt()
            diff > 0.5 -> floored.toInt() + 1
            floored.toInt() % 2 == 0 -> floored.toInt()
            else -> floored.toInt() + 1
        }
    }
}