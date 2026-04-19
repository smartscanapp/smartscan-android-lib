# Indexers Documentation

## ImageIndexer

Processes image media IDs, generates embeddings, and stores them in an embedding store.

Constructor:

```kotlin
ImageIndexer(
    embedder: ImageEmbeddingProvider,
    store: EmbeddingStore,
    maxImageSize: Int = 225,
    context: Context,
    listener: ProcessorListener<Long, StoredEmbedding>? = null,
    memoryOptions: MemoryOptions = MemoryOptions(),
    batchSize: Int = 10
)
```

---

### onProcess(context, item): StoredEmbedding

Processes a single image item ID.

Behavior:

* Resolves `MediaStore.Images.Media` URI from ID
* Loads bitmap with size constraint (`maxImageSize`)
* Generates embedding using `ImageEmbeddingProvider`
* Wraps result into `StoredEmbedding`

Output:

* `StoredEmbedding(id, date, embedding)`

---

### onBatchComplete(context, batch)

Behavior:

* Persists batch embeddings into `EmbeddingStore`
* Forwards batch event to listener if present

---

### Design Notes

* Optimized for batch execution via `BatchProcessor`
* Embeddings are generated per media item ID from `MediaStore`
* Storage is decoupled from indexing logic

---

## VideoIndexer

Processes video media IDs, extracts representative frames, and generates prototype embeddings.

Constructor:

```kotlin
VideoIndexer(
    embedder: ImageEmbeddingProvider,
    frameCount: Int = 10,
    width: Int,
    height: Int,
    context: Context,
    listener: ProcessorListener<Long, StoredEmbedding>? = null,
    batchSize: Int = 10,
    memoryOptions: MemoryOptions = MemoryOptions(),
    store: EmbeddingStore
)
```

---

### onProcess(context, item): StoredEmbedding

Processes a single video item ID.

Behavior:

* Resolves `MediaStore.Video.Media` URI from ID
* Extracts frames from video
* Resizes frames to `(width, height)`
* Embeds frames in batch
* Aggregates embeddings into a single prototype vector

Output:

* `StoredEmbedding(id, date, prototypeEmbedding)`

---

### onBatchComplete(context, batch)

Behavior:

* Persists batch embeddings into `EmbeddingStore`
* Forwards batch event to listener if present

---

### Design Notes

* Frame-based video representation using sampled keyframes
* Aggregates multiple frame embeddings into a single vector
* Higher compute cost due to frame extraction and batch embedding
* Intended for semantic video retrieval and clustering pipelines
