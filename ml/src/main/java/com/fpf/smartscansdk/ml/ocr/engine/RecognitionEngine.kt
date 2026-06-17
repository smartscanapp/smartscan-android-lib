package com.fpf.smartscansdk.ml.ocr.engine

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import com.fpf.smartscansdk.core.copyFloatBuffer
import com.fpf.smartscansdk.ml.models.ModelAssetSource
import com.fpf.smartscansdk.ml.models.OnnxModel
import com.fpf.smartscansdk.ml.models.loaders.FileLoader
import com.fpf.smartscansdk.ml.models.loaders.ResourceLoader
import com.fpf.smartscansdk.ml.ocr.postprocess.CTCDecoder
import com.fpf.smartscansdk.ml.ocr.preprocess.RecPreprocessor
import java.nio.FloatBuffer

internal class RecognitionEngine(
    context: Context,
    modelSource: ModelAssetSource,
    private val characterList: List<String>,
) {
    data class RecognitionResult(
        val texts: List<Pair<String, Float>>,
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

    fun recognize(crops: List<Bitmap>): RecognitionResult {
        val preStart = System.currentTimeMillis()
        val preResult = RecPreprocessor.preprocessBatch(crops)
        val preprocessMs = System.currentTimeMillis() - preStart

        val infStart = System.currentTimeMillis()
        val (outputData, outputShape) = run(preResult.tensorData, preResult.shape)
        val inferenceMs = System.currentTimeMillis() - infStart

        val postStart = System.currentTimeMillis()
        val decoded = CTCDecoder.decode(outputData, outputShape, characterList)
        val postprocessMs = System.currentTimeMillis() - postStart

        val inputShape = preResult.shape.map { it.toInt() }
        val timeMs = preprocessMs + inferenceMs + postprocessMs

        return RecognitionResult(
            texts = decoded,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs,
            timeMs = timeMs,
            inputShape = inputShape,
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