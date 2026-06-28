package com.fpf.smartscansdk.core.cluster

import com.fpf.smartscansdk.core.embeddings.Embedding

data class ClusterMetadata(
    var prototypeSize: Int,
    var meanSimilarity: Float = 0f,
    var stdSimilarity: Float = 0f,
    var label: String? = null
)

data class Cluster(
    val clusterId: ClusterId,
    var embedding: Embedding,
    var metadata: ClusterMetadata,
)
