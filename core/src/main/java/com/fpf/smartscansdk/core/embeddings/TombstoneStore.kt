package com.fpf.smartscansdk.core.embeddings

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.math.max
import kotlin.math.roundToInt

class TombstoneStore(
    private val file: File,
    private val minTombstonesBeforeCompact: Int = 100,
    private val tombstoneRatioLimit: Float = 0.1f
) {

    private var cache: MutableSet<Long>? = null

    val exists: Boolean
        get() = file.exists()


    fun read(): Set<Long> {
        cache?.let { return it }

        val ids = mutableSetOf<Long>()

        if (file.exists()) {
            DataInputStream(
                BufferedInputStream(
                    FileInputStream(file)
                )
            ).use { input ->
                while (input.available() >= Long.SIZE_BYTES) {
                    ids += input.readLong()
                }
            }
        }

        cache = ids
        return ids
    }

    fun append(ids: List<Long>) {
        if (ids.isEmpty()) return

        val tombstones = read() as MutableSet<Long>
        val newIds = ids.filter { tombstones.add(it) }
        if (newIds.isEmpty()) return

        file.parentFile?.mkdirs()

        DataOutputStream(
            BufferedOutputStream(
                FileOutputStream(file, true)
            )
        ).use { output ->
            newIds.forEach(output::writeLong)
        }
    }

    fun clear() {
        cache = mutableSetOf()
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

    fun recoverIfNeeded(ids: List<Long>) {
        val tombstones = read().toMutableSet()

        if (!tombstones.removeAll(ids.toSet())) {
            return
        }
        writeReplace(tombstones)
    }

    private fun writeReplace(ids: Set<Long>) {
        if (ids.isEmpty()) {
            clear()
            return
        }

        file.parentFile?.mkdirs()

        val tempFile = File(file.parentFile, "${file.name}.tmp")

        DataOutputStream(
            BufferedOutputStream(
                FileOutputStream(tempFile)
            )
        ).use { output ->
            ids.forEach(output::writeLong)
        }

        Files.move(
            tempFile.toPath(),
            file.toPath(),
            StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE
        )
    }
}