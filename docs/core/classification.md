# Classification

## ClassificationResult

Represents the output of a classification operation.

```kotlin
data class ClassificationResult(
  val classId: Long?,
  val similarity: Float
)
```

Fields:

* `classId: Long?` — identifier of the predicted class, or `null` if no class matches
* `similarity: Float` — similarity score of the selected class

---

## fewShotClassify

Performs prototype-based few-shot classification using clustered embeddings.

```kotlin
fun fewShotClassify(
    embedding: Embedding,
    classPrototypes: List<Cluster>
): ClassificationResult
```

### Parameters

* `embedding: Embedding` — embedding to classify
* `classPrototypes: List<Cluster>` — prototype clusters representing each class

---

### Behavior

Computes the best matching class prototype for the input embedding.

Process:

* Computes the similarity between the input embedding and each cluster prototype.
* Requires the query embedding and prototype embedding to use the same embedding representation (`Embedding.F32` or `Embedding.QInt8`).
* Computes the acceptance threshold for each prototype as:

  * `meanSimilarity - stdSimilarity`

* Accepts a prototype only if:

  * similarity ≥ threshold
  * similarity is greater than the current best score

* Selects the highest-scoring valid prototype.

---

### Output

Returns:

* `ClassificationResult(classId, similarity)`

Where:

* `classId` is the identifier of the selected cluster, or `null` if no prototype satisfies its threshold.
* `similarity` is the similarity score of the selected prototype.

---

### Error Handling

Throws:

* `InvalidEmbeddingType` if the input embedding type does not match the prototype embedding type.

---

### Design Notes

* Supports both `Embedding.F32` and `Embedding.QInt8`.
* Uses dot-product similarity for comparison.
* Derives classification thresholds from cluster statistics.
* Implements deterministic prototype-based few-shot classification.