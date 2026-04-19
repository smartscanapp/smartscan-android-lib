package com.fpf.smartscansdk.ml.models

enum class ModelType {
    OBJECT_DETECTOR,
    IMAGE_ENCODER,
    TEXT_ENCODER
}

enum class ModelName {
    ALL_MINILM_L6_V2,
    ALL_DISTIL_ROBERTA_V1,
    CLIP_VIT_B_32_TEXT,
    CLIP_VIT_B_32_IMAGE,
    DINOV2_SMALL,
    INCEPTION_RESNET_V1,
    ULTRA_LIGHT_FACE_DETECTOR
}

data class ModelInfo(
    val type: ModelType,
    val url: String,
    val path: String,
    val name: ModelName,
    val resourceFiles: List<String>? = null,
    val hash: String? = null
)