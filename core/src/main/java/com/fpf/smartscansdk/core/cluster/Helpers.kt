package com.fpf.smartscansdk.core.cluster

import com.fpf.smartscansdk.core.embeddings.Embedding
import com.fpf.smartscansdk.core.embeddings.generatePrototypeEmbedding
import com.fpf.smartscansdk.core.embeddings.getSimilarities
import com.fpf.smartscansdk.core.embeddings.getTopN
import kotlin.math.sqrt

fun mergeSimilarClusters(clusterPrototypes: Map<ClusterId, Cluster>, mergeThreshold: Float = 0.9f, ): ClusterMerges {
    val clusterMerges: ClusterMerges = linkedMapOf()
    val clusterIds = clusterPrototypes.keys.toList()
    val clusterEmbeddings = clusterPrototypes.values.map { it.embedding }

    for (idx in clusterEmbeddings.indices) {
        val sims = when(val emb = clusterEmbeddings[idx]){
            is Embedding.F32 -> getSimilarities(emb.vector, clusterEmbeddings.map { (it as Embedding.F32 ).vector}).toMutableList()
            is Embedding.QInt8 -> getSimilarities(emb.vector, clusterEmbeddings.map { (it as Embedding.QInt8).vector }).toMutableList()
        }
        sims[idx] = 0f
        val mergeIndices = getTopN(sims, sims.size, mergeThreshold)

        if (mergeIndices.isNotEmpty()) {
            val merges = mergeIndices.map { j -> clusterIds[j] }
            clusterMerges[clusterIds[idx]] = merges
        }
    }
    return clusterMerges
}

fun computeClusterMetrics(embeddings: List<FloatArray> ): Triple<FloatArray, Float, Float>{
    val prototypeEmbedding = generatePrototypeEmbedding(embeddings)
    val sims = getSimilarities(prototypeEmbedding, embeddings)
    val meanSim = sims.average().toFloat()
    val stdSim = sqrt(sims.map { (it - meanSim) * (it - meanSim) }.average()).toFloat()
    return Triple(prototypeEmbedding, meanSim, stdSim)
}

fun computeClusterMetrics(embeddings: List<ByteArray> ): Triple<ByteArray, Float, Float>{
    val prototypeEmbedding = generatePrototypeEmbedding(embeddings)
    val sims = getSimilarities(prototypeEmbedding, embeddings)
    val meanSim = sims.average().toFloat()
    val stdSim = sqrt(sims.map { (it - meanSim) * (it - meanSim) }.average()).toFloat()
    return Triple(prototypeEmbedding, meanSim, stdSim)
}

fun computeClusterMetrics(embeddings: List<Embedding> ): Triple<Embedding, Float, Float>{
    val prototypeEmbedding = generatePrototypeEmbedding(embeddings)
    val sims = getSimilarities(prototypeEmbedding, embeddings)
    val meanSim = sims.average().toFloat()
    val stdSim = sqrt(sims.map { (it - meanSim) * (it - meanSim) }.average()).toFloat()
    return Triple(prototypeEmbedding, meanSim, stdSim)
}
