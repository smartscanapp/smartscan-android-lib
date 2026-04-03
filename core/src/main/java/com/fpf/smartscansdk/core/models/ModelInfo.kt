package com.fpf.smartscansdk.core.models

enum class ModelType(val tag: String) {
    OBJECT_DETECTOR("object_detector"),
    IMAGE_ENCODER("image_encoder"),
    TEXT_ENCODER("text_encoder")
}

enum class ModelName {
    ALL_MINILM_L6_V2,
    ALL_DISTIL_ROBERTA_V1,
    CLIP_VIT_B_32_TEXT,
    CLIP_VIT_B_32_IMAGE,
    DINOV2_SMALL
}
data class ModelInfo(
    val type: ModelType,
    val url: String,
    val path: String,
    val name: ModelName,
    val resourceFiles: List<String>? = null,
    val hash: String? = null
)