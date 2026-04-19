package com.fpf.smartscansdk.ml.models

val ModelRegistry  =  mapOf<ModelName, ModelInfo>(
    ModelName.ALL_MINILM_L6_V2 to ModelInfo(
        type= ModelType.TEXT_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/minilm_sentence_transformer_quant_128.zip",
        path="all_minilm_l6_v2",
        name=ModelName.ALL_MINILM_L6_V2,
        resourceFiles=listOf("minilm_sentence_transformer_quant.onnx", "vocab.txt", "config.json")
    ),
    ModelName.ALL_DISTIL_ROBERTA_V1 to ModelInfo(
        type= ModelType.TEXT_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/all_distilroberta_v1_quant.zip",
        path="all_distilroberta_v1",
        name=ModelName.ALL_DISTIL_ROBERTA_V1,
        resourceFiles=listOf("sentence-transformers_all-distilroberta-v1_quant.onnx", "vocab.json", "merges.txt")
    ),
    ModelName.CLIP_VIT_B_32_TEXT to ModelInfo(
        type= ModelType.TEXT_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/clip_text_encoder_quant.zip",
        path="clip_vit_b_32_text",
        name=ModelName.CLIP_VIT_B_32_TEXT,
        resourceFiles=listOf("clip_text_encoder_quant.onnx", "vocab.json", "merges.txt")
    ),
    ModelName.CLIP_VIT_B_32_IMAGE to ModelInfo(
        type= ModelType.IMAGE_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/clip_image_encoder_quant.onnx",
        path="clip_image_encoder_quant.onnx",
        name= ModelName.CLIP_VIT_B_32_IMAGE,
    ),
    ModelName.DINOV2_SMALL to ModelInfo(
        type= ModelType.IMAGE_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.0/dinov2_small_quant.onnx",
        path="dinov2_small_quant.onnx",
        name=ModelName.DINOV2_SMALL,
        ),
    ModelName.INCEPTION_RESNET_V1 to ModelInfo(
        type= ModelType.IMAGE_ENCODER,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/inception_resnet_v1_quant.onnx",
        path="inception_resnet_v1_quant.onnx",
        name=ModelName.INCEPTION_RESNET_V1,
    ),
    ModelName.ULTRA_LIGHT_FACE_DETECTOR to ModelInfo(
        type= ModelType.OBJECT_DETECTOR,
        url="https://github.com/dev-diaries41/smartscan-models/releases/download/1.0.1/ultra_light_face_detector.onnx",
        path="ultra_light_face_detector.onnx",
        name=ModelName.ULTRA_LIGHT_FACE_DETECTOR,
    ),
)