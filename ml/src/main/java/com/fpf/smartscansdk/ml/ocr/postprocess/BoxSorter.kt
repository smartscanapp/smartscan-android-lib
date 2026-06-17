package com.fpf.smartscansdk.ml.ocr.postprocess

import com.fpf.smartscansdk.ml.ocr.model.OCRBox
import kotlin.math.abs


object BoxSorter {
    private const val ROW_THRESHOLD_Y = 10f

    fun sortInReadingOrder(boxes: List<OCRBox>): List<OCRBox> {
        if (boxes.size <= 1) return boxes

        val list = boxes.toMutableList()
        list.sortWith(compareBy({ it.points[0].y }, { it.points[0].x }))

        // Align with PaddleX SortQuadBoxes: bubble left-to-right inside a 10px row band.
        for (i in 0 until list.size - 1) {
            var j = i
            while (j >= 0) {
                val next = list[j + 1]
                val curr = list[j]
                if (abs(next.points[0].y - curr.points[0].y) < ROW_THRESHOLD_Y &&
                    next.points[0].x < curr.points[0].x
                ) {
                    list[j] = next
                    list[j + 1] = curr
                    j--
                } else {
                    break
                }
            }
        }
        return list
    }
}
