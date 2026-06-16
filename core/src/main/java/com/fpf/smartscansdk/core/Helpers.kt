package com.fpf.smartscansdk.core

import java.nio.FloatBuffer

fun copyFloatBuffer(buffer: FloatBuffer): FloatArray {
    val duplicate = buffer.duplicate()
    duplicate.rewind()
    val output = FloatArray(duplicate.remaining())
    duplicate.get(output)
    return output
}