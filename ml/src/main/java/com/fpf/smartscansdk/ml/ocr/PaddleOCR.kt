package com.fpf.smartscansdk.ml.ocr

import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.ocr.engine.OCREngine
import com.fpf.smartscansdk.ml.ocr.engine.OCREngineResult
import com.fpf.smartscansdk.ml.ocr.model.OCRRunResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PaddleOCR private constructor(
    private val engine: OCREngine,
) {
    companion object {

        suspend fun create(
            context: Context,
            config: PaddleOCRConfig,
            detModelAssetSource: ModelAssetSource,
            recModelAssetSource: ModelAssetSource,
            recConfigAssetSource: ModelAssetSource,
        ): PaddleOCR {
            val appContext = context.applicationContext
            return withContext(Dispatchers.IO) {
                val engine = OCREngine(
                    appContext,
                    detModelAsset = detModelAssetSource,
                    recModelAsset = recModelAssetSource,
                    recConfigAsset = recConfigAssetSource,
                    config = config
                )
                PaddleOCR(engine)
            }
        }
    }

    fun isInitialized() = engine.isInitialized()

    suspend fun initialize() = engine.initialize()

    suspend fun recognize(bitmap: Bitmap): OCRRunResult {
        if (bitmap.width == 0 || bitmap.height == 0) {
            throw SmartScanException.InvalidInput("Invalid image input for OCR")
        }
        return recognizeResult { engine.run(bitmap) }
    }

    suspend fun recognize(imageBytes: ByteArray): OCRRunResult {
        if (imageBytes.isEmpty()) {
            throw SmartScanException.InvalidInput("Invalid image input for OCR")
        }
        return recognizeResult { engine.run(imageBytes) }
    }

    fun close() = engine.close()

    private suspend fun recognizeResult(runEngine: () -> OCREngineResult): OCRRunResult {
        return withContext(Dispatchers.IO) {
            val result = runEngine()
            OCRRunResult(
                results = result.results,
                detectionTimeMs = result.detectionTimeMs,
                recognitionTimeMs = result.recognitionTimeMs,
                totalTimeMs = result.totalTimeMs,
                lineCount = result.lineCount,
                detPreprocessMs = result.detPreprocessMs,
                detInferenceMs = result.detInferenceMs,
                detPostprocessMs = result.detPostprocessMs,
                recPreprocessMs = result.recPreprocessMs,
                recInferenceMs = result.recInferenceMs,
                recPostprocessMs = result.recPostprocessMs,
                pipelineOverheadMs = result.pipelineOverheadMs,
                coldLoadTimeMs = result.coldLoadTimeMs,
                detInputShape = result.detInputShape,
                recInputShapes = result.recInputShapes,
                perLineRecMs = result.perLineRecMs,
            )
        }
    }

}
