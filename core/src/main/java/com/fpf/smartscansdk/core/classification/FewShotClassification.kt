package com.fpf.smartscansdk.core.classification

import com.fpf.smartscansdk.core.cluster.Cluster
import com.fpf.smartscansdk.core.cluster.ClusterId
import com.fpf.smartscansdk.core.embeddings.Embedding
import com.fpf.smartscansdk.core.embeddings.dot


fun fewShotClassify(embedding: Embedding, classPrototypes: List<Cluster>): ClassificationResult{
    var bestSim = -1f
    var bestPrototype: ClusterId? = null

    for(classPrototype in classPrototypes) {
        val sim = when(embedding){
            is Embedding.F32 -> {
                require(classPrototype.embedding is Embedding.F32){"Embedding mismatch: expected F32"}
                (classPrototype.embedding as Embedding.F32).vector dot embedding.vector
            }
            is Embedding.QInt8 -> {
                require(classPrototype.embedding is Embedding.QInt8){"Embedding mismatch: expected QInt8"}
                (classPrototype.embedding as Embedding.QInt8).vector dot embedding.vector
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