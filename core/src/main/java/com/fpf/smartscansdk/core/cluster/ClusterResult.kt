package com.fpf.smartscansdk.core.cluster

data class ClusterResult(
    val clusters: Map<ClusterId, Cluster>,
    val assignments: Assignments,
    val merges: ClusterMerges?,
)