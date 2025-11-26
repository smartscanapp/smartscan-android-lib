package com.fpf.smartscansdk.core.embeddings

import com.fpf.smartscansdk.core.data.ClassificationError
import com.fpf.smartscansdk.core.data.ClassificationResult
import com.fpf.smartscansdk.core.data.PrototypeEmbedding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.sqrt


infix fun FloatArray.dot(other: FloatArray) = foldIndexed(0.0) { i, acc, cur -> acc + cur * other[i] }.toFloat()

fun normalizeL2(inputArray: FloatArray): FloatArray {
    var norm = 0.0f
    for (i in inputArray.indices) {
        norm += inputArray[i] * inputArray[i]
    }
    norm = sqrt(norm)
    return inputArray.map { it / norm }.toFloatArray()
}

fun getSimilarities(embedding: FloatArray, comparisonEmbeddings: List<FloatArray>): List<Float> {
    return comparisonEmbeddings.map { embedding dot it }
}

fun getTopN(similarities: List<Float>, n: Int, threshold: Float = 0f): List<Int> {
    return similarities.indices.filter { similarities[it] >= threshold }
        .sortedByDescending { similarities[it] }
        .take(n)
}

suspend fun generatePrototypeEmbedding(rawEmbeddings: List<FloatArray>): FloatArray =
    withContext(Dispatchers.Default) {
        if (rawEmbeddings.isEmpty()) throw IllegalStateException("Missing embeddings")
        val embeddingLength = rawEmbeddings[0].size
        val sum = FloatArray(embeddingLength)
        for (emb in rawEmbeddings) for (i in emb.indices) sum[i] += emb[i]

        normalizeL2(FloatArray(embeddingLength) { i -> sum[i] / rawEmbeddings.size })
    }


fun flattenEmbeddings(embeddings: List<FloatArray>, embeddingDim: Int): FloatArray {
    val batchSize = embeddings.size
    val flattened = FloatArray(batchSize * embeddingDim)
    for (i in embeddings.indices) {
        System.arraycopy(embeddings[i], 0, flattened, i * embeddingDim, embeddingDim)
    }
    return flattened
}

fun unflattenEmbeddings(flattened: FloatArray, embeddingDim: Int): List<FloatArray> {
    val batchSize = flattened.size / embeddingDim
    val embeddings = mutableListOf<FloatArray>()
    for (i in 0 until batchSize) {
        val embedding = FloatArray(embeddingDim)
        System.arraycopy(flattened, i * embeddingDim, embedding, 0, embeddingDim)
        embeddings.add(embedding)
    }
    return embeddings
}

fun fewShotClassify(embedding: FloatArray, classPrototypes: List<PrototypeEmbedding>, threshold: Float = 0.4f, confidenceMargin: Float = 0.05f ): ClassificationResult{
    if(classPrototypes.size < 2) return ClassificationResult.Failure(error= ClassificationError.MINIMUM_CLASS_SIZE) // Using a single class prototype leads to many false positives

    // No threshold filter applied here to allow confidence check by comparing top 2 matches
    // A bigger margin of confidence indicates a strong match, which helps reduce false positives
    val similarities = getSimilarities(embedding, classPrototypes.map { it.embeddings })

    val top2 = getTopN(similarities, 2)
    val bestIndex = top2[0]
    val bestSim = similarities[bestIndex]
    val secondSim = top2.getOrNull(1)?.let { similarities[it] } ?: 0f

    if (bestSim < threshold) return ClassificationResult.Failure(error= ClassificationError.THRESHOLD)
    if((bestSim - secondSim) < confidenceMargin) return ClassificationResult.Failure(error= ClassificationError.CONFIDENCE_MARGIN) //inconclusive -  gap between best and second is too small

    val classId = classPrototypes[bestIndex].id
    return ClassificationResult.Success(classId=classId, similarity = bestSim)
}


