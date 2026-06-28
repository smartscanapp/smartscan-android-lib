package com.fpf.smartscansdk.core.embeddings

sealed interface Embedding {
    val size: Int
        get() = when(this){
            is F32 -> this.vector.size
            is QInt8 -> this.vector.size
        }
    data class F32(val vector: FloatArray): Embedding
    data class QInt8(val vector: ByteArray): Embedding

}

fun Embedding.F32.toQInt8(): Embedding.QInt8 = Embedding.QInt8(this.vector.toQInt8())
fun Embedding.QInt8.toF32(): Embedding.F32 = Embedding.F32(this.vector.toF32())
fun Embedding.toFloatArray(): FloatArray = when(this){
    is Embedding.F32 -> this.vector
    is Embedding.QInt8 -> this.toF32().vector
}

data class StoredEmbedding(
    val id: Long,
    val date: Long,
    val embedding: Embedding
)