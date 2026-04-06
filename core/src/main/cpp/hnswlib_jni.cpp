//
// Created by d41dev on 05/04/2026.
//

#include <jni.h>
#include "hnswlib/hnswlib.h"

using namespace hnswlib;

static HierarchicalNSW<float>* index = nullptr;

extern "C" {

JNIEXPORT void JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_initIndex(
    JNIEnv* env,
    jobject,
    jint dim,
    jint maxElements,
    jint efConstruction,
    jint m
) {
    index = new HierarchicalNSW<float>(new InnerProductSpace(dim), maxElements, m, efConstruction);
}

JNIEXPORT void JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_setEf(
    JNIEnv*,
    jobject,
    jint ef
) {
    index->ef_ = ef;
}

JNIEXPORT void JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_addItem(
    JNIEnv* env,
    jobject,
    jfloatArray vector,
    jint id
) {
    jfloat* data = env->GetFloatArrayElements(vector, nullptr);
    index->addPoint((void*)data, id);
    env->ReleaseFloatArrayElements(vector, data, 0);
}

JNIEXPORT jintArray JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_knnQuery(
    JNIEnv* env,
    jobject,
    jfloatArray vector,
    jint k
) {
    jfloat* data = env->GetFloatArrayElements(vector, nullptr);

    auto result = index->searchKnn((void*)data, k);

    jintArray output = env->NewIntArray(k);
    jint* out = env->GetIntArrayElements(output, nullptr);

    int i = 0;
    while (!result.empty() && i < k) {
        out[i++] = result.top().second;
        result.pop();
    }

    env->ReleaseFloatArrayElements(vector, data, 0);
    env->ReleaseIntArrayElements(output, out, 0);

    return output;
}

// Save index to file
JNIEXPORT void JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_saveIndex(
        JNIEnv* env,
        jobject,
        jstring path
) {
    const char* c_path = env->GetStringUTFChars(path, nullptr);
    if (index != nullptr) {
        index->saveIndex(c_path);
    }
    env->ReleaseStringUTFChars(path, c_path);
}

// Load index from file
JNIEXPORT void JNICALL
Java_com_fpf_smartscansdk_core_embeddings_HnswIndex_loadIndex(
        JNIEnv* env,
        jobject,
        jstring path,
        jint dim
) {
    const char* c_path = env->GetStringUTFChars(path, nullptr);

    // Delete old index if exists
    if (index != nullptr) {
        delete index;
        index = nullptr;
    }
    // Create a space object for the index
    SpaceInterface<float>* space = new InnerProductSpace(dim);

    // Create an empty index (maxElements = 0 is okay here)
    index = new HierarchicalNSW<float>(space, 0);

    // Load the index from file, providing the space pointer
    index->loadIndex(c_path, space);
    env->ReleaseStringUTFChars(path, c_path);
}
}


