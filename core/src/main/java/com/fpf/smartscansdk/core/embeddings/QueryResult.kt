package com.fpf.smartscansdk.core.embeddings

data class QueryResult(
    val ids: List<Long> = emptyList(),
    val sims: List<Float>? = null
)