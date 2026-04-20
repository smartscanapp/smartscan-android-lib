## ModelManager

Handles lifecycle of ML models on device: download, import, validation, listing, deletion, and model instantiation.

---

## ROOT_DIR

`filesDir/models/`

---

## downloadModelInternal

```kotlin
suspend fun downloadModelInternal(
    context: Context,
    modelInfo: ModelInfo,
    onProgress: ((Int) -> Unit)? = null
): Result<File>
```

Downloads a model into internal storage.

* Streams file from network to local storage
* Reports progress (0–100) if available
* Uses temporary file then atomic rename
* Returns final model file

---

## downloadModelExternal

```kotlin
fun downloadModelExternal(context: Context, url: String)
```

Opens external browser for model download URL.

---

## importModel

```kotlin
suspend fun importModel(context: Context, modelInfo: ModelInfo, uri: Uri)
```

Imports a model from a URI into internal storage.

* Copies URI content into model directory
* Replaces existing model if present

---

## modelExists

```kotlin
fun modelExists(context: Context, model: ModelName): Boolean
```

Checks whether a model is fully available locally.

---

## deleteModel

```kotlin
fun deleteModel(context: Context, modelInfo: ModelInfo): Boolean
```

Deletes a model from internal storage.

---

## listModels

```kotlin
fun listModels(context: Context, type: ModelType? = null): List<ModelName>
```

Returns installed models, optionally filtered by type.

---

## getModelFile

```kotlin
fun getModelFile(context: Context, modelInfo: ModelInfo): File
```

Resolves model path:

```
filesDir/models/<modelInfo.path>
```

---

## downloadFileInternal

Core network download utility.

* Streams HTTP response to local file
* Writes to temp file first
* Reports progress if content length is known
* Replaces final file atomically

---

## Model Providers

### Text Embedders

```kotlin
fun getTextEmbedder(context: Context, modelName: ModelName): TextEmbeddingProvider
```

Returns initialized text embedding model.

Supported models:

* `ALL_MINILM_L6_V2`
* `CLIP_VIT_B_32_TEXT`

Throws `ModelNotDownloaded` if missing.

---

### Image Embedders

```kotlin
fun getImageEmbedder(context: Context, modelName: ModelName): ImageEmbeddingProvider
```

Supported models:

* `DINOV2_SMALL`
* `CLIP_VIT_B_32_IMAGE`
* `INCEPTION_RESNET_V1`

Throws `ModelNotDownloaded` if missing.

---

### Object Detectors

```kotlin
fun getObjectDetector(context: Context, modelName: ModelName): DetectorProvider
```

Supported models:

* `ULTRA_LIGHT_FACE_DETECTOR`

Throws `ModelNotDownloaded` if missing.

---

## Design Notes

* Internal storage only (`filesDir/models`)
* Atomic file replacement for safety
* Progress-aware streaming downloads
* Model validation enforced before instantiation
* Strict model-type matching for provider factories
