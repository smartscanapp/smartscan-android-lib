package com.fpf.smartscansdk.core.embeddings

import android.app.Application
import android.content.Context
import com.fpf.smartscansdk.core.processors.BatchProcessor
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val QUANT_SCALE: Int = 127
infix fun FloatArray.dot(other: FloatArray) = foldIndexed(0.0) { i, acc, cur -> acc + cur * other[i] }.toFloat()
infix fun ByteArray.dot(other: ByteArray) = foldIndexed(0.0) { i, acc, cur -> acc + cur * other[i] }.toFloat() / (QUANT_SCALE * QUANT_SCALE).toFloat()

fun FloatArray.toQInt8(): ByteArray {
    val q = ByteArray(size)
    for (i in indices) {
        q[i] = (this[i] * QUANT_SCALE.toFloat()).roundToInt().toByte()
    }
    return q
}

fun ByteArray.toF32(): FloatArray {
    val x = FloatArray(size)
    for (i in indices) {
        x[i] = this[i].toFloat() / QUANT_SCALE.toFloat()
    }
    return x
}

fun normalizeL2(rawEmbed: FloatArray): FloatArray {
    var norm = 0.0f
    for (i in rawEmbed.indices) {
        norm += rawEmbed[i] * rawEmbed[i]
    }
    norm = sqrt(norm)
    return rawEmbed.map { it / norm }.toFloatArray()
}

fun normalizeL2(rawEmbed: ByteArray): ByteArray {
    val inputArray = rawEmbed.toF32()
    var norm = 0.0f
    for (i in inputArray.indices) {
        norm += inputArray[i] * inputArray[i]
    }
    norm = sqrt(norm)
    return inputArray.map { it / norm }.toFloatArray().toQInt8()
}

fun getSimilarities(embedding: FloatArray, comparisonEmbeddings: List<FloatArray>): List<Float> {
    return comparisonEmbeddings.map { embedding dot it }
}

fun getSimilarities(embedding: ByteArray, comparisonEmbeddings: List<ByteArray>): List<Float> {
    return comparisonEmbeddings.map { embedding dot it }
}

fun getSimilarities(embedding: Embedding, comparisonEmbeddings: List<Embedding>): List<Float> {
    return comparisonEmbeddings.map {
        when(embedding){
            is Embedding.F32 -> embedding.vector dot it.toF32Embed().vector
            is Embedding.QInt8 -> embedding.vector dot it.toQInt8Embed().vector
        }
    }
}
fun getTopN(similarities: List<Float>, n: Int, threshold: Float = 0f): List<Int> {
    return similarities.indices.filter { similarities[it] >= threshold }
        .sortedByDescending { similarities[it] }
        .take(n)
}

fun generatePrototypeEmbedding(embeddings: List<FloatArray>): FloatArray{
        val embeddingLength = embeddings[0].size
        val sum = FloatArray(embeddingLength)
        for (emb in embeddings) for (i in emb.indices) sum[i] += emb[i]
        return normalizeL2(FloatArray(embeddingLength) { i -> sum[i] / embeddings.size })
    }

fun generatePrototypeEmbedding(embeddings: List<ByteArray>): ByteArray{
    val embeddingLength = embeddings[0].size
    val sum = FloatArray(embeddingLength)
    for (emb in embeddings){
        val dequantEmb = emb.toF32()
        for (i in dequantEmb.indices) sum[i] += dequantEmb[i]
    }
    return normalizeL2(FloatArray(embeddingLength) { i -> sum[i] / embeddings.size }).toQInt8()
}

fun generatePrototypeEmbedding(embeddings: List<Embedding>): Embedding{
    val embeddingLength = embeddings[0].size
    val sum = FloatArray(embeddingLength)
    for (emb in embeddings){
        val floatArray = emb.toF32()
        for (i in floatArray.indices) sum[i] += floatArray[i]
    }
    val norm = normalizeL2(FloatArray(embeddingLength) { i -> sum[i] / embeddings.size })
    val isQuant = embeddings.first() is Embedding.QInt8
    return if(isQuant) norm.toQInt8Embed() else norm.toF32Embed()
}



// updatedPrototype = ((N * currentPrototype) + sum(newEmbedding)) / (N + newN)
fun updatePrototypeEmbedding(embedding: FloatArray, newEmbeddings: List<FloatArray>, currentN: Int): Pair<FloatArray, Int> {
    val updatedN = currentN + newEmbeddings.size
    val sumNew = sumEmbeddings(newEmbeddings)
    val updatedPrototype = FloatArray(embedding.size)
    if(currentN > 0){
        for(i in updatedPrototype.indices) updatedPrototype[i] = currentN.toFloat() * embedding[i]
    }
    for (i in updatedPrototype.indices) updatedPrototype[i] += sumNew[i]
    for (i in updatedPrototype.indices) updatedPrototype[i] /= updatedN.toFloat()
    return Pair(normalizeL2(updatedPrototype), updatedN)
}

fun updatePrototypeEmbedding(embedding: ByteArray, newEmbeddings: List<ByteArray>, currentN: Int): Pair<ByteArray, Int> {
    val updatedN = currentN + newEmbeddings.size
    val embAsFloatArr = embedding.toF32()
    val sumNew = sumEmbeddings(newEmbeddings)
    val updatedPrototype = FloatArray(embAsFloatArr.size)
    if(currentN > 0){
        for(i in updatedPrototype.indices) updatedPrototype[i] = currentN.toFloat() * embAsFloatArr[i]
    }
    for (i in updatedPrototype.indices) updatedPrototype[i] += sumNew[i]
    for (i in updatedPrototype.indices) updatedPrototype[i] /= updatedN.toFloat()
    return Pair(normalizeL2(updatedPrototype).toQInt8(), updatedN)
}

fun updatePrototypeEmbedding(embedding: Embedding, newEmbeddings: List<Embedding>, currentN: Int): Pair<Embedding, Int> {
    val updatedN = currentN + newEmbeddings.size
    val sumNew = sumEmbeddings(newEmbeddings)
    val updatedPrototype = FloatArray(embedding.size)
    val embAsFloatArr = embedding.toF32()

    if(currentN > 0){
        for(i in updatedPrototype.indices) updatedPrototype[i] = currentN.toFloat() * embAsFloatArr[i]
    }
    for (i in updatedPrototype.indices) updatedPrototype[i] += sumNew[i]
    for (i in updatedPrototype.indices) updatedPrototype[i] /= updatedN.toFloat()
    val isQuant = embedding is Embedding.QInt8
    val norm = normalizeL2(updatedPrototype)
    val embed =  if(isQuant) norm.toQInt8Embed() else norm.toF32Embed()
    return Pair(embed, updatedN)
}

fun flattenEmbeddings(embeddings: List<FloatArray>, embeddingDim: Int): FloatArray {
    val batchSize = embeddings.size
    val flattened = FloatArray(batchSize * embeddingDim)
    for (i in embeddings.indices) {
        System.arraycopy(embeddings[i], 0, flattened, i * embeddingDim, embeddingDim)
    }
    return flattened
}

fun flattenEmbeddings(embeddings: List<ByteArray>, embeddingDim: Int): ByteArray {
    val batchSize = embeddings.size
    val flattened = ByteArray(batchSize * embeddingDim)
    for (i in embeddings.indices) {
        System.arraycopy(embeddings[i], 0, flattened, i * embeddingDim, embeddingDim)
    }
    return flattened
}

fun unflattenEmbeddings(flattened: FloatArray, embeddingDim: Int): List<FloatArray> {
    val batchSize = flattened.size / embeddingDim
    val embeddings = mutableListOf<FloatArray>()
    for (i in 0 until batchSize) {
        val embedding = FloatArray(embeddingDim)
        System.arraycopy(flattened, i * embeddingDim, embedding, 0, embeddingDim)
        embeddings.add(embedding)
    }
    return embeddings
}

fun unflattenEmbeddings(flattened: ByteArray, embeddingDim: Int): List<ByteArray> {
    val batchSize = flattened.size / embeddingDim
    val embeddings = mutableListOf<ByteArray>()
    for (i in 0 until batchSize) {
        val embedding = ByteArray(embeddingDim)
        System.arraycopy(flattened, i * embeddingDim, embedding, 0, embeddingDim)
        embeddings.add(embedding)
    }
    return embeddings
}

@JvmName("sumEmbeddingsFloat")
fun sumEmbeddings(embeddings: List<FloatArray>): FloatArray {
    val sum = FloatArray(embeddings[0].size)
    for (emb in embeddings) {
        for (i in emb.indices) {
            sum[i] += emb[i]
        }
    }
    return sum
}

@JvmName("sumEmbeddingsByte")
fun sumEmbeddings(embeddings: List<ByteArray>): FloatArray {
    val sum = FloatArray(embeddings[0].size)
    for (emb in embeddings) {
        val floatArray = emb.toF32()
        for (i in floatArray.indices) {
            sum[i] += floatArray[i]
        }
    }
    return sum
}

@JvmName("sumEmbeddings")
fun sumEmbeddings(embeddings: List<Embedding>): FloatArray {
    val sum = FloatArray(embeddings[0].size)
    for (emb in embeddings) {
        val floatArray = emb.toF32()
        for (i in floatArray.indices) {
            sum[i] += floatArray[i]
        }
    }
    return sum
}

suspend fun <T>embedBatch(context: Context, embedder: EmbeddingProvider<T>, data: List<T>): List<FloatArray> {
    val allEmbeddings = mutableListOf<FloatArray>()

    val processor = object : BatchProcessor<T, FloatArray>(context = context.applicationContext as Application) {
        override suspend fun onProcess(context: Context, item: T): FloatArray {
            return embedder.embed(item)
        }
        override suspend fun onBatchComplete(context: Context, batch: List<FloatArray>) {
            allEmbeddings.addAll(batch)
        }
    }

    processor.run(data)
    return allEmbeddings
}


