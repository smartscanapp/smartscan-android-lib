package com.fpf.smartscansdk.ml.models.loaders

interface IModelLoader<T> {
    suspend fun load(): T
}