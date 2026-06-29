package com.fpf.smartscansdk.core

sealed class SmartScanException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class InvalidEmbeddingStoreFile(message: String = "Invalid embedding store file", val count: Int? = null, val fileSize: Int? = null, val expectedFileSize: Int? = null) : SmartScanException(message)
    class CodecMismatch(message: String = "Codec mismatch") : SmartScanException(message)

    class InvalidEmbeddingType(message: String = "Embedding type mismatch") : SmartScanException(message)

    class InvalidEmbeddingDimension(message: String = "Embedding dimension mismatch") : SmartScanException(message)

    class ModelNotInitialised(message: String = "Model not initialised") : SmartScanException(message)

    class InvalidTextEmbedderResourceFiles(message: String = "Resources files must be of the same type (LocalFile or Resource") : SmartScanException(message)

    class InvalidModelFile(message: String = "Invalid model file") : SmartScanException(message)

    class ModelDownloadFailed(message: String = "Failed to download model") : SmartScanException(message)

    class InvalidModelType(message: String = "Invalid model type") : SmartScanException(message)

    class ModelNotDownloaded(message: String = "Model not downloaded") : SmartScanException(message)

    class ConfigParseFailed(message: String = "Failed to parse config", cause: Throwable? = null) : SmartScanException(message, cause)
    class InvalidInput(message: String = "Input is empty or invalid") : SmartScanException(message)
    class InferenceFailed(message: String = "Inference failed", cause: Throwable? = null) : SmartScanException(message, cause)

}