package com.fpf.smartscansdk.ml.providers.embeddings

import com.fpf.smartscansdk.core.data.ClassificationError
import com.fpf.smartscansdk.core.data.ClassificationResult
import com.fpf.smartscansdk.core.data.PrototypeEmbedding
import com.fpf.smartscansdk.core.embeddings.getSimilarities
import com.fpf.smartscansdk.core.embeddings.getTopN


fun classify(embedding: FloatArray, classPrototypes: List<PrototypeEmbedding>, threshold: Float = 0.4f, confidenceMargin: Float = 0.05f ): ClassificationResult{
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


