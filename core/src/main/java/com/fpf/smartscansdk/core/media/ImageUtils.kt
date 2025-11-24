package com.fpf.smartscansdk.core.media

import android.content.Context
import android.graphics.*
import android.net.Uri
import androidx.core.graphics.scale
import kotlin.math.max
import kotlin.math.min

fun centerCrop(bitmap: Bitmap, imageSize: Int): Bitmap {
    val cropX: Int
    val cropY: Int
    val cropSize: Int
    if (bitmap.width >= bitmap.height) {
        cropX = bitmap.width / 2 - bitmap.height / 2
        cropY = 0
        cropSize = bitmap.height
    } else {
        cropX = 0
        cropY = bitmap.height / 2 - bitmap.width / 2
        cropSize = bitmap.width
    }
    var bitmapCropped = Bitmap.createBitmap(
        bitmap, cropX, cropY, cropSize, cropSize
    )
    bitmapCropped = bitmapCropped.scale(imageSize, imageSize, false)
    return bitmapCropped
}

fun getScaledDimensions(width: Int, height: Int, maxSize: Int = 1024): Pair<Int, Int> {
    if (width <= maxSize && height <= maxSize) {
        return width to height
    }
    return if (width >= height) {
        val scale = maxSize.toFloat() / width
        maxSize to (height * scale).toInt()
    } else {
        val scale = maxSize.toFloat() / height
        (width * scale).toInt() to maxSize
    }
}

fun getBitmapFromUri(context: Context, uri: Uri, maxSize: Int): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        val (w, h) = getScaledDimensions(info.size.width, info.size.height, maxSize)
        decoder.setTargetSize(w, h)
    }.copy(Bitmap.Config.ARGB_8888, true)
}


fun nms(boxes: List<FloatArray>, scores: List<Float>, iouThreshold: Float): List<Int> {
    if (boxes.isEmpty()) return emptyList()

    val indices = scores.indices.sortedByDescending { scores[it] }.toMutableList()
    val keep = mutableListOf<Int>()

    while (indices.isNotEmpty()) {
        val current = indices.removeAt(0)
        keep.add(current)
        val currentBox = boxes[current]

        indices.removeAll { idx ->
            val iou = computeIoU(currentBox, boxes[idx])
            iou > iouThreshold
        }
    }
    return keep
}

private fun computeIoU(boxA: FloatArray, boxB: FloatArray): Float {
    val x1 = max(boxA[0], boxB[0])
    val y1 = max(boxA[1], boxB[1])
    val x2 = min(boxA[2], boxB[2])
    val y2 = min(boxA[3], boxB[3])
    val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
    val areaA = max(0f, boxA[2] - boxA[0]) * max(0f, boxA[3] - boxA[1])
    val areaB = max(0f, boxB[2] - boxB[0]) * max(0f, boxB[3] - boxB[1])
    val unionArea = areaA + areaB - intersectionArea
    return if (unionArea <= 0f) 0f else intersectionArea / unionArea
}