package com.fpf.smartscansdk.ml.ocr.postprocess

import android.graphics.PointF

internal object QuadGeometry {

    fun orderMinAreaRectPoints(points: Array<PointF>): List<PointF> {
        val sorted = points.sortedBy { it.x }

        val left0 = sorted[0]
        val left1 = sorted[1]
        val right0 = sorted[2]
        val right1 = sorted[3]

        val (topLeft, bottomLeft) =
            if (left1.y > left0.y) left0 to left1 else left1 to left0

        val (topRight, bottomRight) =
            if (right1.y > right0.y) right0 to right1 else right1 to right0

        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }
}