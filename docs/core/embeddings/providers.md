# Embedding Provider Documentation

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
