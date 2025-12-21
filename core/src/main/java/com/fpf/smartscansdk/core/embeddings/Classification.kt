package com.fpf.smartscansdk.core.embeddings

data class ClassificationResult(val classId: Long?, val similarity: Float )

fun fewShotClassify(embedding: FloatArray, classPrototypes: List<Embedding>, classPrototypeCohesionScore: Map<Long, Float>): ClassificationResult{
    var bestSim = -1f
    var bestPrototype: Embedding? = null

    for(classPrototype in classPrototypes) {
        val sim = classPrototype.embeddings dot embedding
        val cohesionScore = classPrototypeCohesionScore[classPrototype.id]
        if(cohesionScore != null && sim >= cohesionScore && sim > bestSim){
            bestSim = sim
            bestPrototype = classPrototype
        }
    }
    val classId = bestPrototype?.id
    return ClassificationResult(classId=classId, similarity = bestSim)
}