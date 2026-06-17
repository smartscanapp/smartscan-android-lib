package com.fpf.smartscansdk.ml.ocr.postprocess

import android.graphics.PointF
import kotlin.math.*

internal object PolygonUnclip {
    private const val EPS = 1e-6
    private const val ARC_TOLERANCE = 0.25

    fun unclip(points: List<PointF>, unclipRatio: Float): List<PointF> {
        if (points.size < 3) return points

        val signedArea = signedArea(points)
        val area = abs(signedArea)
        val perimeter = perimeter(points)

        if (!area.isFinite() || !perimeter.isFinite() || area <= EPS || perimeter <= EPS) {
            return points
        }

        val distance = area * unclipRatio / perimeter
        if (!distance.isFinite() || distance <= EPS) return points

        val clockwiseInImageCoords = signedArea > 0.0

        val normals = ArrayList<PointF>(points.size)
        for (i in points.indices) {
            val start = points[i]
            val end = points[(i + 1) % points.size]

            val dx = end.x - start.x
            val dy = end.y - start.y
            val len = hypot(dx, dy)
            if (!len.isFinite() || len <= EPS) return points

            val normal = if (clockwiseInImageCoords) {
                PointF((dy / len).toFloat(), (-dx / len).toFloat())
            } else {
                PointF((-dy / len).toFloat(), (dx / len).toFloat())
            }

            normals.add(normal)
        }

        val expanded = ArrayList<PointF>()

        for (i in points.indices) {
            appendRoundJoin(
                output = expanded,
                center = points[i],
                fromNormal = normals[(i - 1 + normals.size) % normals.size],
                toNormal = normals[i],
                distance = distance,
                clockwiseInImageCoords = clockwiseInImageCoords
            )
        }

        return if (expanded.size >= 3) expanded else points
    }

    private fun appendRoundJoin(
        output: MutableList<PointF>,
        center: PointF,
        fromNormal: PointF,
        toNormal: PointF,
        distance: Double,
        clockwiseInImageCoords: Boolean,
    ) {
        var startAngle = atan2(fromNormal.y.toDouble(), fromNormal.x.toDouble())
        var endAngle = atan2(toNormal.y.toDouble(), toNormal.x.toDouble())

        if (clockwiseInImageCoords) {
            while (endAngle < startAngle) endAngle += 2.0 * PI
        } else {
            while (endAngle > startAngle) endAngle -= 2.0 * PI
        }

        val sweep = endAngle - startAngle
        val steps = ceil(abs(sweep) / arcStepAngle(distance)).toInt().coerceAtLeast(1)

        for (step in 0..steps) {
            val angle = startAngle + sweep * step.toDouble() / steps.toDouble()
            appendPoint(
                output,
                PointF(
                    (center.x + cos(angle) * distance).toFloat(),
                    (center.y + sin(angle) * distance).toFloat()
                )
            )
        }
    }

    private fun arcStepAngle(distance: Double): Double {
        val ratio = (1.0 - ARC_TOLERANCE / distance).coerceIn(-1.0, 1.0)
        val step = 2.0 * acos(ratio)
        return if (step.isFinite() && step > EPS) step else PI / 8.0
    }

    private fun appendPoint(output: MutableList<PointF>, point: PointF) {
        val last = output.lastOrNull()
        if (last == null || hypot(point.x - last.x, point.y - last.y) > EPS) {
            output.add(point)
        }
    }

    private fun signedArea(points: List<PointF>): Double {
        var sum = 0.0
        for (i in points.indices) {
            val p1 = points[i]
            val p2 = points[(i + 1) % points.size]
            sum += p1.x * p2.y - p2.x * p1.y
        }
        return sum / 2.0
    }

    private fun perimeter(points: List<PointF>): Double {
        var length = 0.0
        for (i in points.indices) {
            val p1 = points[i]
            val p2 = points[(i + 1) % points.size]
            length += hypot(p2.x - p1.x, p2.y - p1.y)
        }
        return length
    }
}