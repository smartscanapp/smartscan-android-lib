package com.fpf.smartscansdk.core.embeddings

// Minimal JNI-based wrapper for hnswlib (cosine space)
// Requires you to build hnswlib with JNI bindings

class HnswIndex(
    private val dim: Int,
    private val maxElements: Int = 1_000_000,
    private val efConstruction: Int = 200,
    private val m: Int = 16,
    private val efSearch: Int = 50,
) {
    companion object {
        init {
            System.loadLibrary("hnswlib_jni") // name of compiled .so/.dll
        }
    }

    private var initialized = false

    external fun initIndex(
        dim: Int,
        maxElements: Int,
        efConstruction: Int,
        m: Int
    )

    external fun setEf(ef: Int)

    external fun addItem(vector: FloatArray, id: Int)

    external fun knnQuery(vector: FloatArray, k: Int): IntArray

    external fun saveIndex(path: String)

    external fun loadIndex(path: String, dim: Int)

    fun init() {
        if (!initialized) {
            initIndex(dim, maxElements, efConstruction, m)
            setEf(efSearch)
            initialized = true
        }
    }

    fun add(id: Int, vector: FloatArray) {
        addItem(vector, id)
    }

    fun query(vector: FloatArray, k: Int): List<Int> {
        return knnQuery(vector, k).toList()
    }
}
