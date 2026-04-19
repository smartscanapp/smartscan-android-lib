## Cluster Documentation

## Overview

Incremental clustering system based on embedding similarity and approximate nearest neighbor (HNSW) search. Supports dynamic cluster creation, assignment, updating, and optional merging.

---

## ClusterMetadata

Stores statistical information about a cluster.

* `prototypeSize: Int` — number of items in cluster
* `meanSimilarity: Float` — running mean similarity to prototype
* `stdSimilarity: Float` — similarity standard deviation
* `label: String?` — optional label

---

## Cluster

Represents a cluster.

* `prototypeId: Long` — unique cluster identifier
* `embedding: FloatArray` — centroid/prototype vector
* `metadata: ClusterMetadata` — cluster statistics

---

## ClusterResult

Output of clustering.

* `clusters: Map<ClusterId, Cluster>` — final clusters
* `assignments: Assignments` — item → cluster mapping
* `merges: ClusterMerges?` — optional merge relationships

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

### Purpose

Online clustering using cosine similarity, adaptive thresholds, HNSW approximate nearest neighbor search, and optional voting-based fallback.

---

### Construction

```kotlin
IncrementalClusterer(
    existingClusters: Map<ClusterId, Cluster>? = null,
    defaultThreshold: Float = 0.3f,
    minClusterSize: Int = 2,
    topK: Int = 5
)
```

* `existingClusters` — optional initial clusters
* `defaultThreshold` — baseline similarity threshold
* `minClusterSize` — threshold for adaptive behavior
* `topK` — neighbors used for voting fallback

---

### Key Methods

#### cluster(items: List<StoredEmbedding>): ClusterResult

Processes embeddings incrementally and returns clustering results.

Behavior:

* builds ANN index (HNSW)
* assigns each item to best matching cluster
* applies adaptive thresholding
* falls back to voting when needed
* creates new clusters if required
* updates cluster prototypes incrementally
* optionally performs cluster merging

---

#### clear()

Resets all clusters, assignments, and ANN state.

---

## Assignment Strategy

1. **Direct assignment**
   Applied when similarity exceeds adaptive threshold

2. **Voting fallback**
   Uses top-K nearest neighbors and majority cluster vote

3. **New cluster creation**
   Used when no existing cluster matches criteria

---

## Thresholding

Adaptive threshold is derived from:

* cluster size
* mean similarity
* similarity variance
* global cohesion statistics

It blends a baseline threshold with statistical adjustments to stabilize clustering across varying densities.

---

## Prototype Updates

Each assignment updates:

* centroid embedding (incremental mean)
* mean similarity
* similarity variance

---

## Cluster Merging

Optional post-processing step that merges clusters based on global similarity cohesion criteria.
