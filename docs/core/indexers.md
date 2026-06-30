````md
# Indexers Documentation

## ImageIndexer

Processes image media IDs, generates embeddings, and stores them in an embedding store.

Constructor:

```kotlin
ImageIndexer(
    embedder: ImageEmbeddingProvider,
    store: EmbeddingStore,
    maxImageSize: Int = 225,
    quantize: Boolean = false,
    context: Context,
    listener: ProcessorListener<Long, Pair<Long, Embedding>>? = null,
    memoryOptions: MemoryOptions = MemoryOptions(),
    batchSize: Int = 10
)
```

---

### onProcess(context, item): Pair<Long, Embedding>

Processes a single image media ID.

Behavior:

* Resolves the `MediaStore.Images.Media` URI from the media ID.
* Loads the bitmap using the configured maximum image size.
* Generates an embedding using the `ImageEmbeddingProvider`.
* Returns either an `Embedding.F32` or `Embedding.QInt8` depending on the `quantize` setting.

Output:

* `Pair<Long, Embedding>` containing the media ID and generated embedding.

---

### onBatchComplete(context, batch)

Behavior:

* Retrieves the `DATE_ADDED` timestamp for each processed image.
* Creates `StoredEmbedding` instances for the batch.
* Persists the embeddings using the configured `EmbeddingStore`.
* Forwards the batch event to the listener, if present.

---

### Design Notes

* Built on `BatchProcessor`.
* Reads images directly from `MediaStore`.
* Supports optional embedding quantization.
* Storage is decoupled from embedding generation.

---

## VideoIndexer

Processes video media IDs, generates prototype embeddings from sampled frames, and stores them in an embedding store.

Constructor:

```kotlin
VideoIndexer(
    embedder: ImageEmbeddingProvider,
    frameCount: Int = 10,
    width: Int,
    height: Int,
    quantize: Boolean = false,
    context: Context,
    listener: ProcessorListener<Long, Pair<Long, Embedding>>? = null,
    batchSize: Int = 10,
    memoryOptions: MemoryOptions = MemoryOptions(),
    store: EmbeddingStore
)
```

---

### onProcess(context, item): Pair<Long, Embedding>

Processes a single video media ID.

Behavior:

* Resolves the `MediaStore.Video.Media` URI from the media ID.
* Extracts representative video frames.
* Resizes frames to the configured dimensions.
* Generates embeddings for each frame.
* Produces a prototype embedding from the frame embeddings.
* Returns either an `Embedding.F32` or `Embedding.QInt8` depending on the `quantize` setting.

Output:

* `Pair<Long, Embedding>` containing the media ID and generated prototype embedding.

---

### onBatchComplete(context, batch)

Behavior:

* Retrieves the `DATE_ADDED` timestamp for each processed video.
* Creates `StoredEmbedding` instances for the batch.
* Persists the embeddings using the configured `EmbeddingStore`.
* Forwards the batch event to the listener, if present.

---

### Design Notes

* Built on `BatchProcessor`.
* Uses sampled video frames to represent video content.
* Aggregates frame embeddings into a single prototype embedding.
* Supports optional embedding quantization.
* Frame extraction may fail for videos with unsupported codecs.
* Intended for semantic retrieval, clustering, and classification pipelines.
````
