package com.fpf.smartscansdk.core

sealed class SmartScanException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class CorruptedEmbeddingStoreFile(message: String = "Embedding store file is corrupted") :
        SmartScanException(message)

    class InvalidEmbeddingDimension(message: String = "Embedding dimension mismatch") :
        SmartScanException(message)

    class ModelNotInitialised(message: String = "Model not initialised") :
        SmartScanException(message)

    class InvalidTextEmbedderResourceFiles(message: String = "Resources files must be of the same type (LocalFile or Resource") :
        SmartScanException(message)

    class InvalidModelFile(message: String = "Invalid model file") :
        SmartScanException(message)

    class ModelDownloadFailed(message: String = "Failed to download model") :
        SmartScanException(message)

    class InvalidModelType(message: String = "Invalid model type") :
        SmartScanException(message)

    class ModelNotDownloaded(message: String = "Model not downloaded") :
        SmartScanException(message)
}