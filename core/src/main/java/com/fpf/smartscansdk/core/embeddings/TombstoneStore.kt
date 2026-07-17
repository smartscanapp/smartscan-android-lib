package com.fpf.smartscansdk.core.embeddings

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

class TombstoneStore(
    private val file: File,
    private val minTombstonesBeforeCompact: Int = 100,
    private val tombstoneRatioLimit: Float = 0.1f
) {
    val exists: Boolean
        get() = file.exists()

    fun append(ids: List<Long>) {
        if (ids.isEmpty()) return

        file.parentFile?.mkdirs()

        DataOutputStream(
            BufferedOutputStream(
                FileOutputStream(file, true)
            )
        ).use { output ->
            for (id in ids) {
                output.writeLong(id)
            }
        }
    }

    fun read(): Set<Long> {
        if (!file.exists()) return emptySet()

        val ids = mutableSetOf<Long>()

        DataInputStream(
            BufferedInputStream(
                FileInputStream(file)
            )
        ).use { input ->
            while (input.available() >= Long.SIZE_BYTES) {
                ids.add(input.readLong())
            }
        }

        return ids
    }

    fun clear() {
        if (file.exists()) {
            file.delete()
        }
    }

    fun shouldCompact(tombstoneCount: Int, activeSize: Int): Boolean {
        if (tombstoneCount == 0 || activeSize == 0) return false

        val dynamicLimit = (tombstoneRatioLimit * activeSize).roundToInt()
        val limit = max(dynamicLimit, minTombstonesBeforeCompact)

        return tombstoneCount >= limit
    }
}