package com.fpf.smartscansdk.core.cluster

typealias ItemId = Long
typealias ClusterId = Long
typealias Assignments = MutableMap<ItemId, ClusterId>
typealias MergeId = Long
typealias TargetClusters = List<ClusterId>
typealias ClusterMerges = MutableMap<MergeId, TargetClusters>

