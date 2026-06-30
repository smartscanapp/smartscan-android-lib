# Embedding Storage and Querying Documentation

## EmbeddingStore

Interface for embedding persistence and similarity querying.

Fields:

* `exists: Boolean` — indicates whether the backing store exists

Methods:

Methods:

* `add(embeddings: List<StoredEmbedding>): Int` — adds new embeddings, returning the number added
* `update(embeddings: List<StoredEmbedding>): Int` — updates existing embeddings, returning the number updated
* `remove(ids: List<Long>): Int` — removes embeddings by ID, returning the number removed
* `get(): List<StoredEmbedding>` — retrieves all stored embeddings
* `clear()` — clears the in-memory cache and index
* `save()` — persists the current in-memory state to the backing store
* `query(embedding: Embedding, topK: Int, threshold: Float, ids: Set<Long> = emptySet(), startDate: Long? = null, endDate: Long? = null, includeSims: Boolean = false): QueryResult` — performs a similarity search

* Query parameters:
* `embedding: Embedding` — query embedding
* `topK: Int` — maximum number of results to return
* `threshold: Float` — minimum cosine similarity
* `ids: Set<Long>` — optional ID filter
* `startDate: Long?` — optional inclusive lower timestamp bound
* `endDate: Long?` — optional inclusive upper timestamp bound
* `includeSims: Boolean` — includes similarity scores in the result when `true`

Returns:

* `QueryResult`

    * `ids: List<Long>` — matching embedding IDs
    * `sims: List<Float>?` — similarity scores, when requested

Notes:

* Supports optional ID and date-range filtering.
* Similarity scores are omitted unless explicitly requested.
* Designed for disk- or memory-backed implementations.

---

## FileEmbeddingStore

File-backed implementation of `EmbeddingStore`.

Constructor:

```kotlin
FileEmbeddingStore(
    file: File,
    embeddingDimension: Int,
    quantize: Boolean = false
)
```

Behavior:

* Stores embeddings in a binary file.
* Uses either an `F32EmbeddingCodec` or `QInt8EmbeddingCodec` depending on `quantize`.
* Supports lazy loading with optional in-memory caching.
* Validates embedding dimensions and embedding type before storage and querying.
* Uses a mutex to synchronize file operations.

Internal state:

* `cache: LinkedHashMap<Long, StoredEmbedding>` — lazily initialized cache
* `idToFileOffsetIndex: MutableMap<Long, Long>` — maps embedding IDs to file offsets
* `fileMutex: Mutex` — synchronizes file access

---

### add(embeddings): Int

* Appends embeddings that do not already exist.
* Validates embedding dimensions and type.
* Updates the file index.
* Updates the in-memory cache if it has been initialized.

---

### update(embeddings): Int

* Updates existing embeddings in-place.
* Validates embedding dimensions and type.
* Updates cached embeddings if the cache is initialized.

---

### remove(ids): Int

* Removes embeddings from the in-memory cache and file index.
* Changes are persisted when `save()` is called.

---

### get(): List<StoredEmbedding>

* Returns cached embeddings when available.
* Otherwise loads embeddings and the file index from disk.

---

### save()

* Rewrites the embedding store using the current in-memory state.
* Rebuilds the file index.

---

### query(embedding, topK, threshold, ids, startDate, endDate, includeSims)

* Computes cosine similarity against stored embeddings.
* Supports filtering by IDs and timestamp range.
* Returns the top `K` results above the specified similarity threshold.
* Optionally includes similarity scores in the returned `QueryResult`.
* Validates that the query embedding matches the configured embedding type and dimension.

---

### load()

* Loads embeddings and the file index from disk.
* Initializes the in-memory cache.
* Detects invalid file formats and codec mismatches.

---

## HnswIndex

Uses hnswlib ([https://github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib))
JNI-backed approximate nearest neighbor index.

Constructor:

```kotlin
HnswIndex(dim: Int, maxElements: Int = 1_000_000, efConstruction: Int = 200, m: Int = 16, efSearch: Int = 50)
```

Methods:

* `init()`
* `add(id, vector)`
* `query(vector, k): List<Int>`
* `saveIndex(path)`
* `loadIndex(path, dim)`

Notes:

* Requires native `hnswlib_jni`
* Optimized for large-scale similarity search

---