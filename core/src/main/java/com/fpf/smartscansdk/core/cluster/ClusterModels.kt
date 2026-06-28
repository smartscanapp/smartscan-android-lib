package com.fpf.smartscansdk.core.cluster

import com.fpf.smartscansdk.core.embeddings.Embedding

data class ClusterMetadata(
    var prototypeSize: Int,
    var meanSimilarity: Float = 0f,
    var stdSimilarity: Float = 0f,
    var label: String? = null
)

data class Cluster(
    val prototypeId: Long,
    var embedding: Embedding,
    var metadata: ClusterMetadata,
)

typealias ItemId = Long
typealias ClusterId = Long
typealias Assignments = MutableMap<ItemId, ClusterId>
typealias MergeId = Long
typealias TargetClusters = List<ClusterId>
typealias ClusterMerges = MutableMap<MergeId, TargetClusters>

data class ClusterResult(
    val clusters: Map<ClusterId, Cluster>,
    val assignments: Assignments,
    val merges: ClusterMerges?,
)
