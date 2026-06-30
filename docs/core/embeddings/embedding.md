# Embedding Documentation

## Embedding

Sealed interface representing an embedding vector.

Implementations:

### Embedding.F32

Stores an embedding as a 32-bit floating-point vector.

Fields:

* `vector: FloatArray` — embedding values

### Embedding.QInt8

Stores an embedding as a quantized 8-bit integer vector.

Fields:

* `vector: ByteArray` — quantized embedding values

### Common Properties

Available on all implementations:

* `size: Int` — number of dimensions in the embedding

Notes:

* `Embedding.F32` is intended for full-precision embeddings.
* `Embedding.QInt8` provides a more compact representation with reduced storage requirements.

---

## StoredEmbedding

Represents an embedding associated with an identifier and timestamp.

Fields:

* `id: Long` — unique identifier
* `date: Long` — timestamp
* `embedding: Embedding` — embedding vector

---

## Conversion Extensions

Extension functions for converting between embedding representations.

### Embedding

* `toF32Embed(): Embedding.F32` — returns a floating-point embedding. Quantized embeddings are dequantized.
* `toQInt8Embed(): Embedding.QInt8` — returns a quantized embedding. Floating-point embeddings are quantized.
* `toF32(): FloatArray` — returns the embedding as a `FloatArray`.

### FloatArray

* `toF32Embed(): Embedding.F32` — wraps the array as an `Embedding.F32`.
* `toQInt8Embed(): Embedding.QInt8` — quantizes the array and wraps it as an `Embedding.QInt8`.

### ByteArray

* `toQInt8Embed(): Embedding.QInt8` — wraps the array as an `Embedding.QInt8`.
