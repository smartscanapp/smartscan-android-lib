package com.fpf.smartscansdk.ml.models.loaders

interface ModelLoader<T> {
    suspend fun load(): T
}