# Cluster Documentation

## Overview

Incremental clustering system for embedding-based similarity search and online cluster assignment.

---

## ClusterMetadata

Stores statistical information about a cluster.

Fields:

* `prototypeSize: Int` — number of embeddings represented by the prototype
* `meanSimilarity: Float` — running mean similarity to the prototype
* `stdSimilarity: Float` — running similarity standard deviation
* `label: String?` — optional cluster label

---

## Cluster

Represents a cluster.

Fields:

* `clusterId: ClusterId` — unique cluster identifier
* `embedding: Embedding` — cluster prototype embedding
* `metadata: ClusterMetadata` — cluster statistics

Notes:

* Cluster embeddings are internally converted to `Embedding.F32` by `IncrementalClusterer`, as required by the underlying HNSW index.

---

## ClusterResult

Result returned by the clustering process.

Fields:

* `clusters: Map<ClusterId, Cluster>` — resulting clusters
* `assignments: Assignments` — item-to-cluster assignments
* `merges: ClusterMerges?` — optional cluster merge relationships

---

## Type Aliases

* `ItemId = Long`
* `ClusterId = Long`
* `Assignments = MutableMap<ItemId, ClusterId>`
* `MergeId = Long`
* `TargetClusters = List<ClusterId>`
* `ClusterMerges = MutableMap<MergeId, TargetClusters>`

---

## IncrementalClusterer

Performs online clustering using cosine similarity, adaptive thresholding, HNSW approximate nearest neighbour search, and voting-based assignment.

### Constructor

```kotlin
IncrementalClusterer(
    existingClusters: Map<ClusterId, Cluster>? = null,
    defaultThreshold: Float = 0.3f,
    minClusterSize: Int = 2,
    topK: Int = 5
)
```

Parameters:

* `existingClusters` — optional clusters used to initialize the clusterer
* `defaultThreshold` — baseline cosine similarity threshold
* `minClusterSize` — minimum cluster size before adaptive thresholding is applied
* `topK` — number of nearest neighbours used during ANN queries and voting

---

### cluster(items): ClusterResult

```kotlin
cluster(items: Map<ItemId, Embedding>): ClusterResult
```

Processes embeddings incrementally and returns the resulting clusters, assignments, and optional merge information.

---

### clear()

Resets the clustering state.

Behavior:

* Removes all clusters.
* Clears assignments.
* Resets the HNSW index state.

---

## Assignment Strategy

Embeddings are assigned using the following order:

1. Direct similarity assignment using the adaptive threshold.
2. Majority voting across the nearest neighbours returned by the HNSW index.
3. Creation of a new cluster.

---

## Adaptive Thresholding

Cluster assignment thresholds are computed using:

* default threshold
* cluster size
* cluster mean similarity
* cluster similarity standard deviation
* average dataset cohesion

This allows thresholds to become more selective as clusters mature.

---

## Prototype Updates

After each assignment:

* the cluster prototype embedding is updated incrementally
* prototype size is increased
* mean similarity is updated
* similarity standard deviation is updated

---

## Cluster Merging

After clustering completes, similar clusters may be merged using a global similarity threshold derived from overall cluster cohesion statistics.