package com.fpf.smartscansdk.ml.ocr.engine


import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.copyFloatBuffer
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceLoader
import com.fpf.smartscansdk.ml.ocr.PaddleOCRConfig
import com.fpf.smartscansdk.ml.ocr.model.OCRBox
import com.fpf.smartscansdk.ml.ocr.postprocess.DBPostProcessor
import com.fpf.smartscansdk.ml.ocr.preprocess.DetPreprocessResult
import com.fpf.smartscansdk.ml.ocr.preprocess.DetPreprocessor
import java.nio.FloatBuffer

internal class DetectionEngine(
    context: Context,
    modelSource: ModelAssetSource,
    private val config: PaddleOCRConfig,
) {
    data class DetectionResult(
        val boxes: List<OCRBox>,
        val preprocessMs: Long,
        val inferenceMs: Long,
        val postprocessMs: Long,
        val timeMs: Long,
        val inputShape: List<Int>,
    )

    private val model: OnnxModel = when(modelSource) {
        is ModelAssetSource.Resource -> OnnxModel(ResourceLoader(context.resources, modelSource.resId))
        is ModelAssetSource.LocalFile -> OnnxModel(FileLoader(modelSource.file))
    }

    suspend fun initialize() = model.loadModel()

    fun isInitialized() = model.isLoaded()

    fun close() = model.close()

    fun detect(bitmap: Bitmap): DetectionResult {
        return detect {
            DetPreprocessor.preprocess(
                bitmap,
                config.detLimitSideLen,
                config.detLimitType,
                config.detMaxSideLimit,
                config.detImgMode,
            )
        }
    }

    private fun detect(preprocessor: () -> DetPreprocessResult): DetectionResult {
        if (!model.isLoaded()) throw SmartScanException.ModelNotInitialised()

        val preStart = System.currentTimeMillis()
        val preResult = preprocessor()
        val preprocessMs = System.currentTimeMillis() - preStart
        val infStart = System.currentTimeMillis()
        val (outputData, outputShape) = run(preResult.tensorData, preResult.shape)
        val inferenceMs = System.currentTimeMillis() - infStart
        val postStart = System.currentTimeMillis()
        val boxes = DBPostProcessor.process(
            pred = outputData,
            predShape = outputShape,
            thresh = config.detThresh,
            boxThresh = config.detBoxThresh,
            unclipRatio = config.detUnclipRatio,
            maxCandidates = config.detMaxCandidates,
            useDilation = config.detUseDilation,
            scoreMode = config.detScoreMode,
            boxType = config.detBoxType,
            originalH = preResult.originalH,
            originalW = preResult.originalW,
        )
        val postprocessMs = System.currentTimeMillis() - postStart

        return DetectionResult(
            boxes = boxes,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs,
            timeMs = preprocessMs + inferenceMs + postprocessMs,
            inputShape = listOf(1, 3, preResult.shape[2].toInt(), preResult.shape[3].toInt()),
        )
    }

    private fun run(tensorData: FloatArray, shape: LongArray): Pair<FloatArray, LongArray>{
        val inputName = model.getInputNames()!!.first()
        val tensor = OnnxTensor.createTensor(model.getEnv(), FloatBuffer.wrap(tensorData), shape)
        val output = model.run(mapOf(inputName to tensor))
        val outputTensor = output.values.firstOrNull() as? OnnxTensor?: throw Exception("Output is not an ONNX tensor")
        try {
           return  Pair(copyFloatBuffer(outputTensor.floatBuffer), outputTensor.info.shape)
        }finally {
            output.values.forEach { it.close() }
        }
    }

}
