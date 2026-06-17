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

fun getBitmapFromUri(context: Context, uri: Uri, maxSize: Int? = null): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        maxSize?.let {
            val (w, h) = getScaledDimensions(info.size.width, info.size.height, it)
            decoder.setTargetSize(w, h)
        }
    }.copy(Bitmap.Config.ARGB_8888, true)
}

fun cropFaces(bitmap: Bitmap, boxes: List<FloatArray>): List<Bitmap> {
    val faces = mutableListOf<Bitmap>()
    for (box in boxes) {
        val x1 = max(0, box[0].toInt())
        val y1 = max(0, box[1].toInt())
        val x2 = min(bitmap.width, box[2].toInt())
        val y2 = min(bitmap.height, box[3].toInt())
        val width = x2 - x1
        val height = y2 - y1
        if (width > 0 && height > 0) {
            val faceBitmap = Bitmap.createBitmap(bitmap, x1, y1, width, height)
            faces.add(faceBitmap)
        }
    }
    return faces
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

fun drawBoxes(bitmap: Bitmap, boxes: List<FloatArray>, color: Int, margin: Int = 0, strokeWidth: Float = 2f): Bitmap {
    val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(mutableBitmap)

    val paint = Paint().apply {
        this.color = color
        this.strokeWidth = strokeWidth
        this.style = Paint.Style.STROKE
    }

    for (box in boxes) {
        val x1 = max(0, box[0].toInt() -margin)
        val y1 = max(0, box[1].toInt() -margin)
        val x2 = min(mutableBitmap.width, box[2].toInt() + margin)
        val y2 = min(mutableBitmap.height, box[3].toInt() + margin)

        canvas.drawRect(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat(), paint)
    }

    return mutableBitmap
}


fun resizeToMultipleOf32(src: Bitmap, limitSideLen: Int, limitType: String, maxSideLimit: Int): Bitmap {
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

    return src.scale(newW, newH)
}

fun imdecodeBGR(imageBytes: ByteArray): Bitmap {
    val options = BitmapFactory.Options().apply {
        inPreferredConfig = Bitmap.Config.ARGB_8888
        inScaled = false
    }

    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
        ?: throw IllegalArgumentException("Failed to decode image bytes")
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