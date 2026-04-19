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

* `classId`: Identifier of the predicted class (nullable if no match)
* `similarity`: Best similarity score achieved during classification

---

## fewShotClassify

Performs few-shot classification using prototype embeddings and cohesion constraints.

```kotlin
fun fewShotClassify(
    embedding: FloatArray,
    classPrototypes: List<StoredEmbedding>,
    classPrototypeCohesionScores: Map<Long, Float>
): ClassificationResult
```

---

### Behavior

Computes the best matching class prototype for an input embedding.

Process:

* Iterates over all provided class prototypes
* Computes similarity using dot product between input embedding and prototype embedding
* Retrieves cohesion threshold for each prototype from `classPrototypeCohesionScores`
* Accepts a prototype only if:

    * similarity ≥ cohesion threshold
    * similarity is greater than the current best score
* Selects the highest-scoring valid prototype

---

### Output

Returns:

* `ClassificationResult(classId, similarity)`

Where:

* `classId` is the selected prototype ID, or null if no valid match is found
* `similarity` is the highest dot-product similarity observed (or initial value if none matched)

---

### Design Notes

* Uses dot product similarity for embedding comparison
* Cohesion scores act as per-class confidence thresholds
* Implements hard filtering before selection (threshold gating)
* Suitable for few-shot / prototype-based classification systems
* Deterministic: result depends only on input embeddings and thresholds
