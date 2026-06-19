package com.fpf.smartscansdk.ml.ocr.engine

import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.media.imdecodeBGR
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.ocr.PaddleOCRConfig
import com.fpf.smartscansdk.ml.ocr.model.ModelConfig
import com.fpf.smartscansdk.ml.ocr.model.OCRResult
import com.fpf.smartscansdk.ml.ocr.postprocess.BoxSorter
import com.fpf.smartscansdk.ml.ocr.postprocess.QuadTextCrop

internal class OCREngine(
    context: Context,
    detModelAsset: ModelAssetSource,
    recModelAsset: ModelAssetSource,
    recConfigAsset: ModelAssetSource,
    private val config: PaddleOCRConfig,
    ) {

    private val detectionEngine: DetectionEngine
    private val recognitionEngine: RecognitionEngine


    init {
        val recConfig = ModelConfig.parse(context, recConfigAsset)
        detectionEngine = DetectionEngine(context, detModelAsset, config)
        recognitionEngine = RecognitionEngine(context, recModelAsset, recConfig.characterList)
    }

    suspend fun initialize() {
        detectionEngine.initialize()
        recognitionEngine.initialize()
    }

    fun isInitialized() = detectionEngine.isInitialized() && recognitionEngine.isInitialized()

    fun run(bitmap: Bitmap): OCREngineResult {
        return runInternal(bitmap)
    }

    fun run(imageBytes: ByteArray): OCREngineResult {
        val bmp = try {
            imdecodeBGR(imageBytes)
        } catch (t: Throwable) {
            throw SmartScanException.InvalidInput("Invalid image input for OCR")
        }

        if (bmp.width <= 0 || bmp.height <= 0) {
            bmp.recycle()
            throw SmartScanException.InvalidInput("Invalid image input for OCR")
        }

        return try {
            runInternal(bmp)
        } finally {
            bmp.recycle()
        }
    }

    private fun runInternal(srcBitmap: Bitmap): OCREngineResult {
        val totalStart = System.currentTimeMillis()

        val detResult = detectionEngine.detect(srcBitmap)
        val boxes = detResult.boxes

        if (boxes.isEmpty()) {
            val elapsed = System.currentTimeMillis() - totalStart
            return OCREngineResult(
                results = emptyList(),
                detectionTimeMs = detResult.timeMs,
                recognitionTimeMs = 0,
                totalTimeMs = elapsed,
                lineCount = 0,
                detPreprocessMs = detResult.preprocessMs,
                detInferenceMs = detResult.inferenceMs,
                detPostprocessMs = detResult.postprocessMs,
                detInputShape = detResult.inputShape,
            )
        }

        val sortedBoxes = BoxSorter.sortInReadingOrder(boxes)

        var totalRecPreMs = 0L
        var totalRecInfMs = 0L
        var totalRecPostMs = 0L
        var totalRecMs = 0L

        val allResults = mutableListOf<OCRResult>()
        val recInputShapes = mutableListOf<List<Int>>()
        val perLineRecMs = mutableListOf<Long>()
        val batchSize = config.recBatchSize.coerceAtLeast(1)

        var i = 0
        while (i < sortedBoxes.size) {

            val batchCrops = mutableListOf<Bitmap>()
            val batchBoxIndices = mutableListOf<Int>()
            var next = i

            while (next < sortedBoxes.size && batchCrops.size < batchSize) {
                val crop = QuadTextCrop.crop(srcBitmap, sortedBoxes[next])

                if (crop.width > 0 && crop.height > 0) {
                    batchCrops.add(crop)
                    batchBoxIndices.add(next)
                } else {
                    crop.recycle()
                }
                next++
            }

            try {
                if (batchCrops.isNotEmpty()) {
                    val batchResult = recognitionEngine.recognize(batchCrops)

                    totalRecPreMs += batchResult.preprocessMs
                    totalRecInfMs += batchResult.inferenceMs
                    totalRecPostMs += batchResult.postprocessMs
                    totalRecMs += batchResult.timeMs

                    recInputShapes.add(batchResult.inputShape)

                    if (batchSize == 1) {
                        perLineRecMs.add(batchResult.timeMs)
                    }

                    for (j in batchResult.texts.indices) {
                        val boxIdx = batchBoxIndices[j]
                        val (text, confidence) = batchResult.texts[j]

                        if (confidence >= config.recScoreThresh) {
                            allResults.add(
                                OCRResult(
                                    box = sortedBoxes[boxIdx],
                                    text = text,
                                    confidence = confidence,
                                )
                            )
                        }
                    }
                }
            } finally {
                batchCrops.forEach { it.recycle() }
            }

            i = next
        }

        val totalElapsed = System.currentTimeMillis() - totalStart
        val pipelineOverhead = totalElapsed - detResult.timeMs - totalRecMs

        return OCREngineResult(
            results = allResults,
            detectionTimeMs = detResult.timeMs,
            recognitionTimeMs = totalRecMs,
            totalTimeMs = totalElapsed,
            lineCount = allResults.size,
            detPreprocessMs = detResult.preprocessMs,
            detInferenceMs = detResult.inferenceMs,
            detPostprocessMs = detResult.postprocessMs,
            recPreprocessMs = totalRecPreMs,
            recInferenceMs = totalRecInfMs,
            recPostprocessMs = totalRecPostMs,
            pipelineOverheadMs = pipelineOverhead,
            detInputShape = detResult.inputShape,
            recInputShapes = recInputShapes,
            perLineRecMs = perLineRecMs,
        )
    }

    fun close() {
        detectionEngine.close()
        recognitionEngine.close()
    }
}