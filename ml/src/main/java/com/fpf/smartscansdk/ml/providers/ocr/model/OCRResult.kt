package com.fpf.smartscansdk.ml.providers.ocr.model

data class OCRResult(
    val box: OCRBox,
    val text: String,
    val confidence: Float,
    val wordBoxes: List<OCRBox>? = null,
)
