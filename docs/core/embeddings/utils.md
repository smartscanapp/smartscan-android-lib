
# Embedding Utilities Documentation

## Similarity

Utilities for computing vector similarity.

### dot

Computes the dot product between two embeddings.

Overloads:

* `FloatArray.dot(other: FloatArray): Float`
* `ByteArray.dot(other: ByteArray): Float`

Notes:

* `ByteArray` embeddings are automatically dequantized during computation.

---

### getSimilarities

Computes similarity scores between a query embedding and a collection of embeddings.

Overloads:

* `getSimilarities(embedding: FloatArray, comparisonEmbeddings: List<FloatArray>): List<Float>`
* `getSimilarities(embedding: ByteArray, comparisonEmbeddings: List<ByteArray>): List<Float>`
* `getSimilarities(embedding: Embedding, comparisonEmbeddings: List<Embedding>): List<Float>`

Returns:

* Cosine similarity scores for each comparison embedding.

---

### getTopN

Returns the indices of the highest similarity scores.

```kotlin
getTopN(similarities: List<Float>, n: Int, threshold: Float = 0f): List<Int>
```

Behavior:

* Filters scores below `threshold`.
* Sorts remaining scores in descending order.
* Returns the indices of the top `n` results.

---

## Quantization

Utilities for converting between floating-point and quantized embeddings.

### toQInt8

```kotlin
FloatArray.toQInt8(): ByteArray
```

Converts a normalized floating-point embedding to an 8-bit quantized representation.

---

### toF32

```kotlin
ByteArray.toF32(): FloatArray
```

Converts a quantized embedding back to floating-point.

---

## Normalization

### normalizeL2

Normalizes an embedding using the L2 norm.

Overloads:

* `normalizeL2(rawEmbed: FloatArray): FloatArray`
* `normalizeL2(rawEmbed: ByteArray): ByteArray`

Behavior:

* Produces a unit-length embedding suitable for cosine similarity.

---

## Prototype Embeddings

Utilities for generating and updating prototype embeddings.

### generatePrototypeEmbedding

Creates a prototype embedding by averaging and normalizing multiple embeddings.

Overloads:

* `generatePrototypeEmbedding(embeddings: List<FloatArray>): FloatArray`
* `generatePrototypeEmbedding(embeddings: List<ByteArray>): ByteArray`
* `generatePrototypeEmbedding(embeddings: List<Embedding>): Embedding`

---

### updatePrototypeEmbedding

Updates an existing prototype embedding using additional embeddings.

Overloads:

* `updatePrototypeEmbedding(embedding: FloatArray, newEmbeddings: List<FloatArray>, currentN: Int): Pair<FloatArray, Int>`
* `updatePrototypeEmbedding(embedding: ByteArray, newEmbeddings: List<ByteArray>, currentN: Int): Pair<ByteArray, Int>`
* `updatePrototypeEmbedding(embedding: Embedding, newEmbeddings: List<Embedding>, currentN: Int): Pair<Embedding, Int>`

Returns:

* Updated prototype embedding.
* Updated embedding count.

---

## Embedding Collections

Utilities for manipulating collections of embeddings.

### flattenEmbeddings

Flattens a collection of embeddings into a contiguous array.

Overloads:

* `flattenEmbeddings(embeddings: List<FloatArray>, embeddingDim: Int): FloatArray`
* `flattenEmbeddings(embeddings: List<ByteArray>, embeddingDim: Int): ByteArray`

---

### unflattenEmbeddings

Reconstructs embeddings from a flattened array.

Overloads:

* `unflattenEmbeddings(flattened: FloatArray, embeddingDim: Int): List<FloatArray>`
* `unflattenEmbeddings(flattened: ByteArray, embeddingDim: Int): List<ByteArray>`

---

### sumEmbeddings

Computes the element-wise sum of multiple embeddings.

Overloads:

* `sumEmbeddings(embeddings: List<FloatArray>): FloatArray`
* `sumEmbeddings(embeddings: List<ByteArray>): FloatArray`
* `sumEmbeddings(embeddings: List<Embedding>): FloatArray`

---

## Batch Processing

### embedBatch

Embeds a collection of inputs using an `EmbeddingProvider`.

```kotlin
suspend fun <T> embedBatch(
    context: Context,
    embedder: EmbeddingProvider<T>,
    data: List<T>
): List<FloatArray>
```

Behavior:

* Processes inputs in batches.
* Uses `BatchProcessor` internally.
* Returns the generated embeddings in the same order as the input data.
