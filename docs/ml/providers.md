# Providers

All providers implement `EmbeddingProvider<T>`. Both bundled and downloaded models are supported. The recommended way to use downloaded models is via ModelManager.

## Examples

### Downloaded model

```kotlin
val textEmbedder = ModelManager.getTextEmbedder(application, ModelName.ALL_MINILM_L6_V2)
val imageEmbedder = ModelManager.getImageEmbedder(application, ModelName.DINOV2_SMALL)
```

### Bundled model

```kotlin
val textEmbedder = ClipTextEmbedder(application, ModelAssetSource.Resource(R.raw.clip_text_encoder_quant), vocabSource = ModelAssetSource.Resource(R.raw.vocab), mergesSource = ModelAssetSource.Resource(R.raw.merges))
val imageEmbedder = ClipImageEmbedder(application, ModelAssetSource.Resource(R.raw.clip_image_encoder_quant))
```
``
## Supported image embedders

* dinov2_small
* clip_vit_b_32_image
* inception_resnet_v1 (face)

## Supported text embedders

* all_minilm_l6_v2
* clip_vit_b_32_text

