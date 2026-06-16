package com.fpf.smartscansdk.ml.providers.ocr.model

import com.fpf.smartscansdk.ml.models.ModelAssetSource

sealed class OCRError(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class ModelNotFound(assetSource: ModelAssetSource, cause: Throwable? = null) : OCRError("Model not found: $assetSource", cause)
    class ModelLoadFailed(modelName: String, cause: Throwable) : OCRError("Failed to load $modelName model", cause)
    class ConfigParseFailed(cause: Throwable? = null) : OCRError("Failed to parse config", cause)
    class InvalidImage : OCRError("Input image is empty or invalid")
    class InferenceFailed(stage: String, cause: Throwable) : OCRError("Inference failed at stage '$stage'", cause)
    class DecodeError(message: String, cause: Throwable? = null) : OCRError(message, cause)
}
