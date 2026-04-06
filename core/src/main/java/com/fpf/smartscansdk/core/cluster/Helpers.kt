package com.fpf.smartscansdk.core.cluster

import com.fpf.smartscansdk.core.embeddings.getSimilarities
import com.fpf.smartscansdk.core.embeddings.getTopN

fun mergeSimilarClusters(clusterPrototypes: Map<ClusterId, Cluster>, mergeThreshold: Float = 0.9f, ): ClusterMerges {
    val clusterMerges: ClusterMerges = linkedMapOf()
    val clusterIds = clusterPrototypes.keys.toList()
    val clusterEmbeddings = clusterPrototypes.values.map { it.embedding }

    for (idx in clusterEmbeddings.indices) {
        val emb = clusterEmbeddings[idx]
        val sims = getSimilarities(emb, clusterEmbeddings).toMutableList()
        sims[idx] = 0f
        val mergeIndices = getTopN(sims, sims.size, mergeThreshold)

        if (mergeIndices.isNotEmpty()) {
            val merges = mergeIndices.map { j -> clusterIds[j] }
            clusterMerges[clusterIds[idx]] = merges
        }
    }
    return clusterMerges
}
