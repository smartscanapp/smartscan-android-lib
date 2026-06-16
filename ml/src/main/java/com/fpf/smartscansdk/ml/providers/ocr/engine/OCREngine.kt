package com.fpf.smartscansdk.ml.providers.ocr.engine

import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.providers.ocr.EngineConfig
import com.fpf.smartscansdk.ml.providers.ocr.PaddleOCRConfig
import com.fpf.smartscansdk.ml.providers.ocr.model.ModelConfig
import com.fpf.smartscansdk.ml.providers.ocr.model.OCRError
import com.fpf.smartscansdk.ml.providers.ocr.model.OCRResult
import com.fpf.smartscansdk.ml.providers.ocr.postprocess.BoxSorter
import com.fpf.smartscansdk.ml.providers.ocr.postprocess.QuadTextCrop
import com.fpf.smartscansdk.ml.providers.ocr.util.BitmapUtils
import kotlinx.coroutines.runBlocking

class OCREngine(
    context: Context,
    private val config: PaddleOCRConfig,
    engineConfig: EngineConfig,
    detModelAsset: ModelAssetSource,
    recModelAsset: ModelAssetSource,
    recConfigAsset: ModelAssetSource,
) {

    private val ortManager = ORTSessionManager(context, engineConfig)
    private val detectionEngine: DetectionEngine
    private val recognitionEngine: RecognitionEngine

    val coldLoadTimeMs: Long get() = ortManager.coldLoadTimeMs

    init {
        val recConfig = runBlocking {
            try {
                ortManager.loadModels(detModelAsset, recModelAsset)
                ModelConfig.parse(context, recConfigAsset)
            } catch (t: Throwable) {
                ortManager.release()
                throw t
            }
        }

        detectionEngine = DetectionEngine(ortManager, config)
        recognitionEngine = RecognitionEngine(ortManager, recConfig.characterList)
    }

    fun run(bitmap: Bitmap): OCREngineResult {
        return runInternal(bitmap)
    }

    fun run(imageBytes: ByteArray): OCREngineResult {
        val bmp = try {
            BitmapUtils.imdecodeBGR(imageBytes)
        } catch (t: Throwable) {
            throw OCRError.InvalidImage()
        }

        if (bmp.width <= 0 || bmp.height <= 0) {
            bmp.recycle()
            throw OCRError.InvalidImage()
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
                coldLoadTimeMs = ortManager.coldLoadTimeMs,
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
            coldLoadTimeMs = ortManager.coldLoadTimeMs,
            detInputShape = detResult.inputShape,
            recInputShapes = recInputShapes,
            perLineRecMs = perLineRecMs,
        )
    }

    fun release() {
        ortManager.release()
    }
}