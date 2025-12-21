package com.fpf.smartscansdk.core.data.images

import androidx.room.*
import com.fpf.smartscansdk.core.embeddings.StoredEmbedding

@Entity(tableName = "image_embeddings")
data class ImageEmbeddingEntity(
    @PrimaryKey
    val id: Long,     // Mediastore id
    val date: Long,
    val embeddings: FloatArray
)


fun ImageEmbeddingEntity.toEmbedding() = StoredEmbedding(id, date, embeddings)

fun StoredEmbedding.toEntity() = ImageEmbeddingEntity(id, date, embedding)