package com.fpf.smartscansdk.core.cluster

import com.fpf.smartscansdk.core.embeddings.HnswIndex
import com.fpf.smartscansdk.core.embeddings.StoredEmbedding
import com.fpf.smartscansdk.core.embeddings.dot
import com.fpf.smartscansdk.core.embeddings.updatePrototypeEmbedding
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt


class IncrementalClusterer(
    existingClusters: Map<ClusterId, Cluster>? = null,
    private val defaultThreshold: Float = 0.3f,
    private val minClusterSize: Int = 2,
    private val topK: Int = 5,
) {
    private val clusters: MutableMap<ClusterId, Cluster> = existingClusters?.toMutableMap() ?: linkedMapOf()
    private val assignments: Assignments = linkedMapOf()

    // HNSW ANN index
    private lateinit var annIndex: HnswIndex
    private var annInitialized: Boolean = false
    private val idMap: MutableMap<Int, ItemId> = linkedMapOf()
    private val revIdMap: MutableMap<ItemId, Int> = linkedMapOf()
    private var nextIntId: Int = 0

    fun cluster(items: List<StoredEmbedding>): ClusterResult {
        if (items.isEmpty()) return ClusterResult(clusters, assignments, null)

        val allItems: Map<ItemId, FloatArray> = items.associate { it.id to it.embedding }
        val embedDim = items.first().embedding.size
        initAnn(embedDim)

        val minClusterSize = computeMinClusterSize(allItems.size)

        for (itemId in allItems.keys) {
            val emb = allItems[itemId] ?: continue

            if (clusters.isEmpty()) {
                setAndAssign(itemId, emb)
                addToAnn(itemId, emb)
                continue
            }

            // Query ANN
            val nnIds = queryAnn(emb)
            val clusterIds = clusters.keys.toList()
            val cosSims = clusterIds.map { cid -> emb dot clusters[cid]!!.embedding }.toFloatArray()
            val bestIdx = cosSims.indices.maxBy { cosSims[it] }
            val bestCid = clusterIds[bestIdx]
            val bestSim = cosSims[bestIdx]

            val (avgCohesion, _, _) = computeAverageClusterStats()
            val threshold = getThreshold(clusters[bestCid]!!, avgCohesion, minClusterSize)

            if (bestSim >= threshold) {
                updateAndAssign(itemId, emb, bestCid)
            } else {
                val chosenCid = assignByVotes(emb, avgCohesion, minClusterSize, nnIds)
                if (chosenCid != null) updateAndAssign(itemId, emb, chosenCid)
                else setAndAssign(itemId, emb)
            }

            addToAnn(itemId, emb)
        }

        removeSingletons()

        val (avgCohesion, _, avgStd) = computeAverageClusterStats()
        val mergeThreshold = max(defaultThreshold, avgCohesion - avgStd)
        val clusterMerges = mergeSimilarClusters(clusters, mergeThreshold)
        return ClusterResult(clusters, assignments, clusterMerges)
    }

    fun clear() {
        clusters.clear()
        assignments.clear()
        annInitialized = false
        idMap.clear()
        revIdMap.clear()
        nextIntId = 0
    }

    private fun initAnn(dim: Int) {
        if (!annInitialized) {
            annIndex = HnswIndex(dim = dim)
            annIndex.init()
            annInitialized = true
            nextIntId = 0
        }
    }

    private fun addToAnn(itemId: ItemId, embedding: FloatArray) {
        annIndex.add(nextIntId, embedding)
        idMap[nextIntId] = itemId
        revIdMap[itemId] = nextIntId
        nextIntId++
    }

    private fun queryAnn(embedding: FloatArray): List<ItemId> {
        if (!annInitialized || nextIntId == 0) return emptyList()
        val topKQuery = minOf(topK, nextIntId)
        val nnIntIds = annIndex.query(embedding, topKQuery)
        return nnIntIds.mapNotNull { idMap[it] }
    }

    private fun getThreshold(cluster: Cluster, avgCohesion: Float, minClusterSize: Int): Float {
        val sizeFactor = 1f - exp(-cluster.metadata.prototypeSize.toFloat())
        val baseline = defaultThreshold * sizeFactor
        if (cluster.metadata.prototypeSize < minClusterSize || avgCohesion <= 0f) return baseline

        val x = (cluster.metadata.meanSimilarity - avgCohesion) / max(1e-6f, avgCohesion)
        val alpha = 1f / (1f + exp(-x))
        var adaptiveThreshold = max(cluster.metadata.meanSimilarity - cluster.metadata.stdSimilarity, baseline)
        adaptiveThreshold = alpha * adaptiveThreshold + (1f - alpha) * baseline
        return adaptiveThreshold
    }

    private fun setAndAssign(itemId: ItemId, embedding: FloatArray) {
        val prototypeId = generateId()
        val metadata = ClusterMetadata(
            prototypeSize = 1,
            meanSimilarity = defaultThreshold,
            stdSimilarity = 0f,
        )
        val cluster = Cluster(
            prototypeId = prototypeId,
            embedding = embedding,
            metadata = metadata,
        )
        clusters[prototypeId] = cluster
        assignments[itemId] = prototypeId
    }

    private fun updateAndAssign(itemId: ItemId, embedding: FloatArray, clusterId: ClusterId) {
        val cluster = clusters[clusterId] ?: return
        val oldSize = cluster.metadata.prototypeSize
        val oldMeta = cluster.metadata
        val (newEmbedding, updatedN) = updatePrototypeEmbedding(cluster.embedding, listOf(embedding), oldSize)
        val simNew = newEmbedding dot embedding
        val newMean = if (oldSize >= 1) ((oldMeta.meanSimilarity * oldSize + simNew) / (oldSize + 1)) else simNew
        val newStd = if (oldSize > 1) sqrt((((oldSize - 1) * oldMeta.stdSimilarity.pow(2) + (simNew - oldMeta.meanSimilarity) * (simNew - newMean)) / oldSize).toDouble()).toFloat() else 0f

        cluster.embedding = newEmbedding
        cluster.metadata.prototypeSize = updatedN
        cluster.metadata.meanSimilarity = newMean
        cluster.metadata.stdSimilarity = newStd
        assignments[itemId] = clusterId
    }

    private fun assignByVotes(emb: FloatArray, avgCohesion: Float, minClusterSize: Int, nnIds: List<ItemId>): ClusterId? {
        val (voteCounts, voteSims) = tallyVotes(nnIds, emb)
        if (voteCounts.isEmpty()) return null

        val votedCid = selectTopCluster(voteCounts, voteSims)
        val votedCluster = clusters[votedCid] ?: return null
        val nVotes = voteCounts[votedCid] ?: 0
        val requiredVotes = topK / 2
        if (nVotes < requiredVotes) return null

        val simToVoted = emb dot votedCluster.embedding
        val votedThreshold = getThreshold(votedCluster, avgCohesion, minClusterSize)
        return if (simToVoted >= votedThreshold) votedCid else null
    }

    private fun tallyVotes(neighbourIds: List<ItemId>, embedding: FloatArray): Pair<MutableMap<ClusterId, Int>, MutableMap<ClusterId, MutableList<Float>>> {
        val voteCounts: MutableMap<ClusterId, Int> = linkedMapOf()
        val voteSims: MutableMap<ClusterId, MutableList<Float>> = linkedMapOf()
        for (nid in neighbourIds) {
            val cid = assignments[nid] ?: continue
            val cluster = clusters[cid] ?: continue
            voteCounts[cid] = (voteCounts[cid] ?: 0) + 1
            voteSims.getOrPut(cid) { mutableListOf() }.add(embedding dot cluster.embedding)
        }
        return voteCounts to voteSims
    }

    private fun selectTopCluster(voteCounts: Map<ClusterId, Int>, voteSims: Map<ClusterId, List<Float>>): ClusterId {
        val topValue = voteCounts.values.maxOrNull() ?: 0
        val topCids = voteCounts.filterValues { it == topValue }.keys.toList()
        if (topCids.size == 1) return topCids[0]
        return topCids.maxBy { cid -> voteSims[cid]?.average()?.toFloat() ?: 0f }
    }

    private fun generateId(): Long = System.currentTimeMillis() + nextIntId.toLong()

    private fun computeMinClusterSize(totalItems: Int): Int {
        if (totalItems <= 0) return maxOf(2, minClusterSize)
        val adaptive = maxOf(2, sqrt(totalItems.toFloat()).toInt())
        return maxOf(adaptive, minClusterSize)
    }

    private fun computeAverageClusterStats(): Triple<Float, Float, Float> {
        val cohesions = mutableListOf<Float>()
        val clusterSizes = mutableListOf<Int>()
        val stds = mutableListOf<Float>()
        for (cluster in clusters.values) {
            val size = cluster.metadata.prototypeSize
            clusterSizes.add(size)
            if (size > 1) {
                cohesions.add(cluster.metadata.meanSimilarity)
                stds.add(cluster.metadata.stdSimilarity)
            }
        }
        val avgCohesion = if (cohesions.isNotEmpty()) cohesions.average().toFloat() else 0f
        val avgClusterSize = if (clusterSizes.isNotEmpty()) clusterSizes.average().toFloat() else 0f
        val avgStd = if (stds.isNotEmpty()) stds.average().toFloat() else 0f
        return Triple(avgCohesion, avgClusterSize, avgStd)
    }

    private fun removeSingletons() {
        val singletonItems = assignments.filter { (_, cid) -> clusters[cid]?.metadata?.prototypeSize == 1 }
        for ((itemId, cid) in singletonItems) {
            clusters.remove(cid)
            assignments.remove(itemId)
        }
    }
}