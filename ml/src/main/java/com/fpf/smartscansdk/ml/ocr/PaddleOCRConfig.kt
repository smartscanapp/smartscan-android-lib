package com.fpf.smartscansdk.ml.ocr

data class PaddleOCRConfig(
    val detImgMode: String = "BGR",
    val detLimitSideLen: Int = 64,
    val detLimitType: String = "min",
    val detMaxSideLimit: Int = 4000,
    val detThresh: Float = 0.3f,
    val detBoxThresh: Float = 0.6f,
    val detUnclipRatio: Float = 1.5f,
    val detMaxCandidates: Int = 3000,
    val detUseDilation: Boolean = false,
    val detScoreMode: String = "fast",
    val detBoxType: String = "quad",
    val recScoreThresh: Float = 0.0f,
    val recBatchSize: Int = 1,
)
