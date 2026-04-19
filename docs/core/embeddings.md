# Embeddings Documentation

## EmbeddingProvider<T>

Generic interface for producing embeddings from input data.

Fields / behavior:

* `embeddingDim: Int` — output vector size
* `initialize()` — async initialization
* `isInitialized(): Boolean` — initialization state
* `closeSession()` — optional cleanup
* `embed(data: T): FloatArray` — generates embedding


### TextEmbeddingProvider

Specialization of `EmbeddingProvider<String>`.

Fields:

* `maxTokens: Int` — maximum input length in tokens


### ImageEmbeddingProvider

Type alias:

* `EmbeddingProvider<Bitmap>`

---

## StoredEmbedding

Represents a persisted embedding record.

Fields:

* `id: Long` — unique identifier
* `date: Long` — timestamp
* `embedding: FloatArray` — vector representation

---

## EmbeddingStore

Interface for embedding persistence and retrieval.

Fields / behavior:

* `exists: Boolean` — storage availability

Methods:

* `add(embeddings): Int`
* `update(embeddings): Int`
* `remove(ids): Int`
* `get(): List<StoredEmbedding>`
* `clear()`
* `save()`
* `query(embedding, topK, threshold, ids): List<Long>`

Notes:

* Supports optional ID-filtered queries
* Designed for disk or memory-backed implementations

---

## FileEmbeddingStore

File-backed implementation of `EmbeddingStore`.

Constructor:

```kotlin
FileEmbeddingStore(file: File, embeddingDimension: Int)
```

Behavior:

* Binary format with fixed-size records
* Little-endian encoding
* Header stores record count
* Supports lazy loading and in-memory caching

Internal state:

* `cache: LinkedHashMap<Long, StoredEmbedding>`
* `idToFileOffsetIndex: MutableMap<Long, Long>`
* `fileMutex: Mutex`

---

### add(embeddings): Int

* Appends new embeddings to file
* Skips duplicates
* Updates header and index

---

### update(embeddings): Int

* Overwrites existing records in-place using file offsets
* Updates cache if initialized

---

### remove(ids): Int

* Removes entries from in-memory structures only

---

### get(): List<StoredEmbedding>

* Returns cached embeddings if available
* Otherwise loads full file

---

### save()

* Rewrites entire file
* Rebuilds header and index

---

### query(embedding, topK, threshold, ids)

* Computes cosine similarity
* Returns top-K results above threshold
* Optional ID filtering

---

### load()

* Memory-maps file
* Reconstructs cache and index
* Validates file integrity

---

## HnswIndex

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