# ModelManager

---

## Overview

Handles lifecycle of ML models on device: downloading, importing, validating, listing, and deleting. Supports single-file models and zip-based multi-file models with dependency validation.

---

## ROOT_DIR

Base directory for all stored models.

* `"models"` inside `context.filesDir`

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

### Behavior

* Resolves target path via `getModelFile`
* Detects zip models via `resourceFiles`

### Zip flow

* Downloads to temporary `.zip`
* Extracts into target directory
* Validates required files exist
* Deletes extracted files on validation failure

### Single-file flow

* Direct download into target file

### Notes

* Progress reported 0–100 on main thread
* Uses temp file then atomic rename

---

## downloadModelExternal

```kotlin
fun downloadModelExternal(context: Context, url: String)
```

Opens external browser for model URL.

---

## importModel

```kotlin
suspend fun importModel(context: Context, modelInfo: ModelInfo, uri: Uri)
```

Imports model from external URI.

### Behavior

* Zip models:

  * Copy URI → temp zip
  * Extract into model directory
  * Validate required files
  * Delete temp file

* Single-file models:

  * Direct copy to target location

---

## modelExists

```kotlin
fun modelExists(context: Context, model: ModelName): Boolean
```

Checks if model exists locally.

* File model: checks file existence
* Directory model: checks all required resources exist

---

## deleteModel

```kotlin
fun deleteModel(context: Context, modelInfo: ModelInfo): Boolean
```

Deletes model from storage.

* Files: direct delete
* Directories: recursive delete

---

## listModels

```kotlin
fun listModels(context: Context, type: ModelType? = null): List<ModelName>
```

Returns installed models.

* Filters registry by existence
* Optional filtering by model type

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

## unzipFiles

Extracts zip archive into target directory.

* Skips directories
* Writes files directly
* Returns extracted file list

---

## downloadFileInternal

Core HTTP download implementation.

### Behavior

* Streams file via `HttpURLConnection`
* Writes to `.tmp` file
* Tracks progress if content length is available
* Reports progress only on increase
* Executes callbacks on main thread

### Safety

* Uses temp file during download
* Deletes temp file on failure
* Replaces final file atomically via rename

---

## Design Notes

* Supports both single-file and multi-file model packages
* Validates zip contents against expected dependencies
* Uses atomic file replacement to prevent partial model states
* Designed for offline-first model management
* Separates download, import, and storage concerns
* Optimized for Android sandboxed file system constraints
