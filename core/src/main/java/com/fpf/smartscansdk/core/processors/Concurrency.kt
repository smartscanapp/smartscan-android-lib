package com.fpf.smartscansdk.core.processors

sealed interface Concurrency {
    data class Fixed(val concurrency: Int): Concurrency
    data class Dynamic(val calculateConcurrency: () -> Int): Concurrency
}
