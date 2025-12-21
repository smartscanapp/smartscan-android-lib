# Embeddings


## `IEmbeddingProvider<T>`

Generic interface for generating embeddings from data.

| Property / Method           | Description                               |
| --------------------------- | ----------------------------------------- |
| `embeddingDim`              | Dimensionality of the embeddings produced |
| `closeSession()`            | cleanup method                   |
| `embed(data: T)`            | Produces an embedding for a single item   |
| `embedBatch(data: List<T>)` | Produces embeddings for a batch of items  |

**Type Aliases:**

* `TextEmbeddingProvider = IEmbeddingProvider<String>`
* `ImageEmbeddingProvider = IEmbeddingProvider<Bitmap>`

---

## Embedding Stores

### `IEmbeddingStore`

Interface for storing embeddings persistently.

| Property / Method                     | Description                                            |
| ------------------------------------- | ------------------------------------------------------ |
| `exists`                              | Returns true if the store exists                       |
| `isCached`                            | Indicates if embeddings are currently cached in memory |
| `add(newStoredEmbeddings: List<Embedding>)` | Adds new embeddings to the store                       |
| `remove(ids: List<Long>)`             | Removes embeddings by ID                               |
| `get()`                               | Returns all stored embeddings                          |
| `clear()`                             | Clears the in-memory cache (does not delete the store) |

---

### FileEmbeddingStore

Persistent storage for embeddings in a binary file, which implements IEmbeddingStore. Provides a high-performance, file-backed storage and retrieval mechanism, optimized for on-device vector search.

Key features:

* Full embedding index is memory-mapped for fast in-memory access.
* Optional in-memory caching for repeated queries.
* Supports retrieval of single or multiple embeddings by ID.
* Batch writes to prevent memory pressure.
* Recommended for on-device scenarios where Room/DB performance is insufficient (up to 100× faster).

---

### **Constructor**

| Parameter         | Type      | Description                                                            |
| ----------------- | --------- | ---------------------------------------------------------------------- |
| `file`            | `File`    | File to store the embeddings                                           |
| `embeddingLength` | `Int`     | Dimensionality of the stored embeddings                                |
| `useCache`        | `Boolean` | Whether to maintain an in-memory cache of embeddings (default: `true`) |

---

### **Properties**

| Property   | Type      | Description                                   |
| ---------- | --------- | --------------------------------------------- |
| `exists`   | `Boolean` | Checks if the file exists                     |
| `isCached` | `Boolean` | Indicates if the in-memory cache is populated |

---

### **Methods**

| Method                                | Description                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `get()`                               | Loads all embeddings from the file (or cache if available). Returns `List<Embedding>`.                |
| `get(ids: List<Long>)`                | Retrieves a list of embeddings by their IDs.                                                          |
| `get(id: Long)`                       | Retrieves a single embedding by its ID, or `null` if not found.                                       |
| `add(newStoredEmbeddings: List<Embedding>)` | Adds new embeddings to the store. Updates file header, appends entries, and optionally updates cache. |
| `remove(ids: List<Long>)`             | Removes embeddings by their IDs and saves the updated file.                                           |
| `clear()`                             | Clears in-memory cache (does not delete the file).                                                    |

---

### **Behavior Notes**

* Uses little-endian binary encoding.
* Batch writes in chunks of 1000 embeddings to prevent memory pressure.
* Validates `embeddingLength` on every write.
* Supports retrieval of single or multiple embeddings by ID(s).
* Designed for fast in-memory index usage for similarity search.
* Maintains optional in-memory cache for repeated queries.

---

### **Usage Example**

```kotlin
val store = FileEmbeddingStore(
    file = File(context.filesDir, "image_index.bin"),
    embeddingLength = 512
)

// Add embeddings
store.add(listOf(embedding1, embedding2))

// Retrieve all embeddings
val embeddings = store.get()

// Retrieve specific embeddings
val single = store.get(embedding1.id)
val batch = store.get(listOf(embedding1.id, embedding2.id))

// Remove embeddings
store.remove(listOf(embedding1.id))
```

---

## Retrievers

### `IRetriever`

Interface for querying embeddings.

| Method                                                      | Description                                                                      |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `query(embedding: FloatArray, topK: Int, threshold: Float)` | Returns the top-K embeddings most similar to the input that exceed the threshold |

---

### FileEmbeddingRetriever

Retriever for nearest-neighbor queries over a `FileEmbeddingStore`.

### **Constructor**

| Parameter | Type                 | Description                             |
| --------- | -------------------- | --------------------------------------- |
| `store`   | `FileEmbeddingStore` | The file-based embedding store to query |

---

### **Methods**

| Method                                                      | Description                                                                                           |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `query(embedding: FloatArray, topK: Int, threshold: Float)` | Returns the top-K most similar embeddings to the input vector that exceed the similarity `threshold`. |
| `query(start: Int, end: Int)`                               | Returns a slice of embeddings from the last query results (using cached IDs).                         |

---

### **Behavior Notes**

* Loads all embeddings from the `FileEmbeddingStore`.
* Computes cosine similarity (or L2-normalized distance) against stored embeddings.
* Returns a ranked list of `Embedding` objects or an empty list if none meet the threshold.
* Caches the IDs of the last query for fast batched retrieval.
* `query(start, end)` provides efficient pagination of previous results without recalculating similarities.

---

### **Usage Example**

```kotlin
val retriever = FileEmbeddingRetriever(store)

// Query top 5 similar embeddings
val results = retriever.query(inputEmbedding, topK = 5, threshold = 0.5f)

results.forEach { embedding ->
    println("Found embedding with id=${embedding.id}")
}

// Retrieve a slice of the last query
val batch = retriever.query(start = 0, end = 2)
```

---

## **Embedding Utilities**

Helper functions for vector operations, similarity calculations, and batch embedding processing.

### **Functions**

| Function                                                                         | Description                                                                              |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `FloatArray.dot(other: FloatArray)`                                              | Computes the dot product between two embeddings.                                         |
| `normalizeL2(inputArray: FloatArray)`                                            | Returns an L2-normalized version of the input embedding.                                 |
| `getSimilarities(embedding: FloatArray, comparisonEmbeddings: List<FloatArray>)` | Computes similarities between an embedding and a list of embeddings.                     |
| `getTopN(similarities: List<Float>, n: Int, threshold: Float = 0f)`              | Returns the indices of the top-N embeddings exceeding the threshold.                     |
| `generatePrototypeEmbedding(rawEmbeddings: List<FloatArray>)`                    | Computes a single prototype embedding by averaging and normalizing a list of embeddings. |
| `flattenEmbeddings(embeddings: List<FloatArray>, embeddingDim: Int)`             | Converts a list of embeddings into a single flat FloatArray for batch processing.        |
| `unflattenEmbeddings(flattened: FloatArray, embeddingDim: Int)`                  | Converts a flat FloatArray back into a list of embeddings.                               |

---

### **Extending**

To implement a custom file-based embedding solution:

1. Extend `IEmbeddingStore` to manage reading/writing from custom storage formats.
2. Ensure all embeddings are memory-mapped or loaded for efficient retrieval.
3. Implement an `IRetriever` to compute similarities and return top-K matches.
4. Consider optional caching for frequently accessed embeddings.

---