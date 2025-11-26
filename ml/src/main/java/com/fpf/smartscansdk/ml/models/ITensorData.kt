package com.fpf.smartscansdk.ml.models

import ai.onnxruntime.OnnxJavaType
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

sealed interface TensorData {
    val shape: LongArray

    data class FloatBufferTensor(val data: FloatBuffer, override val shape: LongArray) : TensorData
    data class IntBufferTensor(val data: IntBuffer, override val shape: LongArray) : TensorData
    data class LongBufferTensor(val data: LongBuffer, override val shape: LongArray) : TensorData
    data class DoubleBufferTensor(val data: DoubleBuffer, override val shape: LongArray) : TensorData
    data class ShortBufferTensor(val data: ShortBuffer, override val shape: LongArray, val type: OnnxJavaType? = null) : TensorData
    data class ByteBufferTensor(val data: ByteBuffer, override val shape: LongArray, val type: OnnxJavaType) : TensorData
}
