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

data class StoredEmbedding(
    val id: Long,
    val date: Long,
    val embedding: Embedding
)

fun Embedding.toQInt8Embed(): Embedding.QInt8 = when(this){
    is Embedding.F32 -> Embedding.QInt8(this.vector.toQInt8())
    is Embedding.QInt8 -> this
}
fun Embedding.toF32Embed(): Embedding.F32 = when(this){
    is Embedding.F32 -> this
    is Embedding.QInt8 -> Embedding.F32(this.vector.toF32())
}
fun Embedding.toF32(): FloatArray = when(this){
    is Embedding.F32 -> this.vector
    is Embedding.QInt8 -> this.toF32Embed().vector
}

fun FloatArray.toQInt8Embed(): Embedding.QInt8 = Embedding.QInt8(this.toQInt8())

fun FloatArray.toF32Embed(): Embedding.F32 = Embedding.F32(this)

fun ByteArray.toQInt8Embed(): Embedding.QInt8 = Embedding.QInt8(this)


