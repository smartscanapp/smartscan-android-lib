# **SmartScan SDK**

## Table of Contents

* [Overview](#overview)
* [Documentation](docs/README.md)
* [Installation](#installation)
* [Quick Start](#quick-start)

  - [1. Install Core Module](#1-install-core-module)
  - [2. Install ML Module (Optional)](#2-install-ml-module-optional)
* [Design Choices](#design-choices)
  - [Core and ML](#core-and-ml)
  - [Model](#model)
  - [Embedding Storage](#embedding-storage)
    - [Benchmark Summary](#benchmark-summary) 

* [Gradle / Kotlin Setup Notes](#gradle--kotlin-setup-notes)


## **Overview**

SmartScanSdk is an Android library that powers the **SmartScan app**, providing tools for:

* On-device ML inference
* Semantic search
* Indexing
* Embedding storage
* Incremental clustering
* ANN Search / HNSW Index
* Few shot classification
* Image & video processing
* Efficient batch processing
* Model management


> **Note:** The SDK is designed to be flexible, but its primary use is for the SmartScan app and other apps I am developing. It is also subject to rapid experimental changes.

---

## Installation

Add the JitPack repository to your build file (settings.gradle)

```gradle
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

### **1. Install Core Module**

```gradle
implementation("com.github.dev-diaries41.smartscan-sdk:smartscan-core:1.1.0")
```

### **2. Install ML Module (Optional)**

```gradle
implementation("com.github.dev-diaries41.smartscan-sdk:smartscan-ml:1.1.0")
```

> `ml` depends on `core`, so including it is enough if you need both.

---

## Quick Start

Below is information on how to get started with embedding, clustering, indexing, and searching.

### Embeddings

You can use bundled or downloaded models, see [docs](docs/ml/providers.md) for more details.

#### Text Embeddings

Generate vector embeddings from text strings or batches of text for tasks such as semantic search or similarity comparison.

**Usage Example:**

```kotlin
//import com.fpf.smartscansdk.ml.models.providers.embeddings.clip.ClipTextEmbedder

// downloaded model
val textEmbedder = ModelManager.getTextEmbedder(application, ModelName.ALL_MINILM_L6_V2)

// bundled model
val textEmbedder = ClipTextEmbedder(application, ModelAssetSource.Resource(R.raw.clip_text_encoder_quant), vocabSource = ModelAssetSource.Resource(R.raw.vocab), mergesSource = ModelAssetSource.Resource(R.raw.merges))
val text = "Hello smartscan"
val embedding = textEmbedder.embed(text)

```

**Batch Example:**

Specifically designed for large batches

```kotlin
val texts = listOf("first sentence", "second sentence")
val embeddings = embedBatch(context, textEmbedder, texts)
```

---

#### Image Embeddings

Generate vector embeddings from images (as `Bitmap`) for visual search or similarity tasks.

**Usage Example**

```kotlin
//import com.fpf.smartscansdk.ml.models.providers.embeddings.clip.ClipImageEmbedder

// downloaded model
val imageEmbedder = ModelManager.getImageEmbedder(application, ModelName.DINOV2_SMALL)

// bundled model
val imageEmbedder = ClipImageEmbedder(application, ModelAssetSource.Resource(R.raw.clip_image_encoder_quant))

val embedding = imageEmbedder.embed(bitmap)

```


**Batch Example:**

```kotlin
val images = listOf<Bitmap>()
val embeddings = embedBatch(context, imageEmbedder, images)
```

### Indexing

To get started with indexing media quickly, you can use the provided `ImageIndex` and `VideoIndexer` classes as shown below. You can optionally create your own indexers (including for text related data) by extending the `BatchProcessor`. See [docs](docs/core/processors.md) for more details.

#### Image Indexing

Index images to enable similarity search. The index is saved as a binary file and managed with a FileEmbeddingStore.
> **Important**: During indexing the MediaStore Id is used to as the id in the `StoredEmbedding` which is stored. This can later be used for retrieval.


```kotlin
val imageEmbedder = ClipImageEmbedder(context, ResourceId(R.raw.image_encoder_quant_int8))
val imageStore = FileEmbeddingStore(File(context.filesDir, "image_index.bin"), imageEmbedder.embeddingDim) 
val imageIndexer = ImageIndexer(imageEmbedder, context=context, listener = null, store = imageStore) //optionally pass a listener to handle events
val ids = getImageIds() // placeholder function to get MediaStore image ids
imageIndexer.run(ids)
```

#### Video Indexing

Index videos to enable similarity search. The index is saved as a binary file and managed with a FileEmbeddingStore.
> **Important**: During indexing the MediaStore Id is used to as the id in the `StoredEmbedding` which is stored. This can later be used for retrieval.

```kotlin
val imageEmbedder = ClipImageEmbedder(context, ResourceId(R.raw.image_encoder_quant_int8))
val videoStore = FileEmbeddingStore(File(context.filesDir,  "video_index.bin"), imageEmbedder.embeddingDim )
val videoIndexer = VideoIndexer(imageEmbedder, context=context, listener = null, store = videoStore, width = ClipConfig.IMAGE_SIZE_X, height = ClipConfig.IMAGE_SIZE_Y)
val ids = getVideoIds() // placeholder function to get MediaStore video ids
videoIndexer.run(ids)
```

### Searching

Below shows how to search using both text queries and an image. The returns results are List<Embedding>. You can use the id from each one, which corresponds to the MediaStore id, to retrieve the result images.

#### Text-to-Image Search

 ```kotlin
val imageStore = FileEmbeddingStore(File(context.filesDir, "image_index.bin"), imageEmbedder.embeddingDim) 
val query = "my search query"
val embedding = textEmbedder.embed(query)
val topK = 20
val similarityThreshold = 0.2f
val results = imageStore.query(embedding, topK, similarityThreshold) // returns image ids, optionally pass filter ids

```

#### Reverse Image Search

```kotlin
val imageStore = FileEmbeddingStore(File(context.filesDir, "image_index.bin"), imageEmbedder.embeddingDim) 
val embedding = imageEmbedder.embed(bitmap)
val topK = 20
val similarityThreshold = 0.2f
val results = imageStore.query(embedding, topK, similarityThreshold)
```

#### ANN Search (HNSW Index)

```kotlin
val annIndex = HNSWIndex(dim=512)
val query = "my search query"
val embedding = textEmbedder.embed(query)
val topK = 5
val results = annIndex.query(embedding, topK) // returns nearest neighbour indices must map to item id
```

### Clustering

Incremental clustering groups embeddings as they are added see [docs](docs/core/clustering.md) for more details.

```kotlin
val imageStore = FileEmbeddingStore(File(context.filesDir, "image_index.bin"), imageEmbedder.embeddingDim) 
val itemEmbeds = store.get()
val existingClusters: Map<Long, Cluster> = emptyMap() // optionally pass existing clusters
val clusterer = IncrementalClusterer(existingClusters = existingClusters, defaultThreshold = 0.4f)
val result = clusterer.cluster(itemEmbeds)
```
---

## Design Choices

### Core and ML

* **core** → minimal runtime: shared interfaces, data classes, embeddings, media helpers, processor execution, and efficient batch/concurrent processing.
* **ml** → ML infrastructure and models: model loaders, base models, embedding providers (e.g., CLIP), and few-shot classifiers. Optional or experimental ML-related features can be added under `ml/providers`.

---

### Embedding Storage

The SDK only provides a file based implementation of `EmbeddingStore`, `FileEmbeddingStore` (in core) because the following benchmarks below show much better performance for loading embeddings in comparison to Room.

#### **Benchmark Summary**

File-based memory-mapped loading is significantly faster and scales better.

**Real-Life Test Results**

| Embeddings | Room Time (ms) | File Time (ms) |
|------------|----------------|----------------|
| 640        | 1,237.5        | 32.0           |
| 2,450      | 2,737.2        | 135.0          |


**Instrumented Test Benchmarks**

| Embeddings | Room Time (ms) | File Time (ms) |
|------------|----------------|----------------|
| 2,500      | 5,337.50       | 72.05          |
| 5,000      | 8,095.87       | 126.63         |
| 10,000     | 16,420.67      | 236.51         |
| 20,000     | 36,622.81      | 605.51         |
| 40,000     | 89,363.28      | 939.50         |

![SmartScan Load Benchmark](./benchmarks/smartscan-load-benchmark.png)


___

## Gradle / Kotlin Setup Notes

* Java 17 / Kotlin JVM 17
* `compileSdk = 36`, `targetSdk = 34`, `minSdk = 28`
* `core` exposes `androidx.core:core-ktx`
* `ml` depends on `core` and ONNX Runtime

 ---