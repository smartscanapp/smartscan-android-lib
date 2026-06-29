package com.fpf.smartscansdk.core.classification

import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.core.cluster.Cluster
import com.fpf.smartscansdk.core.cluster.ClusterId
import com.fpf.smartscansdk.core.embeddings.Embedding
import com.fpf.smartscansdk.core.embeddings.dot
import com.fpf.smartscansdk.core.embeddings.toF32Embed
import com.fpf.smartscansdk.core.embeddings.toQInt8Embed


fun fewShotClassify(embedding: Embedding, classPrototypes: List<Cluster>): ClassificationResult{
    var bestSim = -1f
    var bestPrototype: ClusterId? = null

    for(classPrototype in classPrototypes) {
        val sim = when(embedding){
            is Embedding.F32 -> {
                if(classPrototype.embedding !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Embedding mismatch: expected F32")
                classPrototype.embedding.toF32Embed().vector dot embedding.vector
            }
            is Embedding.QInt8 -> {
                if(classPrototype.embedding !is Embedding.QInt8) throw SmartScanException.InvalidEmbeddingType("Embedding mismatch: expected QInt8")
                classPrototype.embedding.toQInt8Embed().vector dot embedding.vector
            }
        }
        val threshold = classPrototype.metadata.meanSimilarity - classPrototype.metadata.stdSimilarity
        if(sim >= threshold && sim > bestSim){
            bestSim = sim
            bestPrototype = classPrototype.clusterId
        }
    }
    val classId = bestPrototype
    return ClassificationResult(classId=classId, similarity = bestSim)
}