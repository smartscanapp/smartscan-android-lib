package com.fpf.smartscansdk.core.embeddings

import java.io.File
import java.nio.channels.FileChannel
import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal class F32EmbeddingCodec(
    private val embeddingDimension: Int,
    private val headerSize: Int,
): EmbeddingCodec {

    // Long + Long + FloatArray
    private val recordSize = (8 + 8 )+ embeddingDimension * 4

    override suspend fun writeReplace(embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>, outputFile: File):Unit = withContext(Dispatchers.IO) {
        val tempFile = File.createTempFile(outputFile.nameWithoutExtension, ".tmp")

        FileOutputStream(tempFile).channel.use { channel ->
            writeHeader(channel, embeddings.size)

            val batchSize = 1000
            var index = 0
            var offset = headerSize

            while (index < embeddings.size) {
                val end = minOf(index + batchSize, embeddings.size)
                val batch = embeddings.subList(index, end)

                // Allocate a smaller buffer for this batch
                val batchBuffer = ByteBuffer.allocate(batch.size * recordSize)
                    .order(ByteOrder.LITTLE_ENDIAN)

                for (embedding in batch) {
                    if(embedding.embedding !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Embedding must be of type F32")
                    batchBuffer.putLong(embedding.id)
                    batchBuffer.putLong(embedding.date)
                    for (f in embedding.embedding.vector) {
                        batchBuffer.putFloat(f)
                    }
                    idToFileOffsetIndex[embedding.id] = offset.toLong()
                    offset += recordSize
                }
                batchBuffer.flip()
                channel.write(batchBuffer)
                index = end
            }
        }
        Files.move(
            tempFile.toPath(),
            outputFile.toPath(),
            StandardCopyOption.REPLACE_EXISTING
        )
    }

    override suspend fun read(file: File ): Pair<LinkedHashMap<Long, StoredEmbedding>, MutableMap<Long, Long>> = withContext(Dispatchers.IO) {
        val cacheMap = LinkedHashMap<Long, StoredEmbedding>()
        val idxMap = mutableMapOf<Long, Long>()

        FileInputStream(file).channel.use { ch ->
            val size = ch.size()
            if (size < headerSize) {
                throw SmartScanException.InvalidEmbeddingStoreFile("File too small to contain header")
            }

            val buffer = ch.map(FileChannel.MapMode.READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN)

            val count = buffer.int
            val expectedSize = headerSize.toLong() + count.toLong() * recordSize.toLong()

            if (count < 0 || expectedSize != size) {
                throw SmartScanException.InvalidEmbeddingStoreFile(
                    "Invalid file: count=$count, fileSize=$size, expectedSize=$expectedSize",
                    fileSize = size.toInt(),
                    expectedFileSize = expectedSize.toInt(),
                    count = count
                )
            }

            var offset = headerSize.toLong()

            repeat(count) {
                val id = buffer.long
                val date = buffer.long
                val floats = FloatArray(embeddingDimension)
                val fb = buffer.asFloatBuffer()
                fb.get(floats)
                buffer.position(buffer.position() + embeddingDimension * 4)

                cacheMap[id] = StoredEmbedding(id, date, floats.toF32Embed())
                idxMap[id] = offset
                offset += recordSize.toLong()
            }
        }
        Pair(cacheMap, idxMap)
    }

    override suspend fun append(file: File, embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>): Int = withContext(Dispatchers.IO) {
        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            val fileExistsAndHasContent = channel.size() >= headerSize.toLong()

            val existingCount = if (fileExistsAndHasContent) {
                readHeader(channel)
            } else {
                0
            }

            val newCount = existingCount + embeddings.size

            // Move to the end (append mode) or start of data section if new file
            val nextOffset = if (fileExistsAndHasContent) {
                channel.size()
            } else {
                headerSize.toLong()
            }

            channel.position(nextOffset)

            val targetChunkBytes = 4 * 1024 * 1024
            val chunkCapacity = maxOf(
                recordSize,
                (targetChunkBytes / recordSize).coerceAtLeast(1) * recordSize
            )
            val writeBuffer = ByteBuffer.allocateDirect(chunkCapacity).order(ByteOrder.LITTLE_ENDIAN)

            fun flushBuffer() {
                writeBuffer.flip()
                while (writeBuffer.hasRemaining()) {
                    channel.write(writeBuffer)
                }
                writeBuffer.clear()
            }

            for (embedding in embeddings) {
                if(embedding.embedding !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Embedding must be of type F32")
                if (writeBuffer.remaining() < recordSize) {
                    flushBuffer()
                }

                writeBuffer.putLong(embedding.id)
                writeBuffer.putLong(embedding.date)
                for (f in embedding.embedding.vector) writeBuffer.putFloat(f)
            }

            if (writeBuffer.position() > 0) {
                flushBuffer()
            }

            writeHeader(channel, newCount)
            channel.force(false)

            // update in-memory file offset index for the newly appended entry and cache
            embeddings.forEachIndexed { index, embedding ->
                idToFileOffsetIndex[embedding.id] = nextOffset + (index.toLong() * recordSize)
            }

            embeddings.size
        }
    }

    override suspend fun update(file: File, embeddings: List<StoredEmbedding>, idToFileOffsetIndex: MutableMap<Long, Long>): Int = withContext(Dispatchers.IO) {
        var updatedCount = 0

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            for (emb in embeddings) {
                if(emb.embedding !is Embedding.F32) throw SmartScanException.InvalidEmbeddingType("Embedding must be of type F32")
                val offset = idToFileOffsetIndex[emb.id] ?: continue
                val buf = ByteBuffer.allocate(recordSize).order(ByteOrder.LITTLE_ENDIAN)
                buf.putLong(emb.id)
                buf.putLong(emb.date)
                for (f in emb.embedding.vector) buf.putFloat(f)
                buf.flip()

                channel.position(offset)
                while (buf.hasRemaining()) {
                    channel.write(buf)
                }
                updatedCount++
            }

            channel.force(false)
        }
        updatedCount
    }

    override suspend fun readHeader(channel: FileChannel): Int = withContext(Dispatchers.IO) {
        val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
        channel.position(0)
        val read = channel.read(headerBuf)
        if (read != headerSize) {
            throw SmartScanException.InvalidEmbeddingStoreFile("Failed to read header count")
        }
        headerBuf.flip()

        val size = channel.size()
        val existingCount = headerBuf.int
        val maxCountFromSize = (size / recordSize)
        if (existingCount !in 0..maxCountFromSize) {
            throw SmartScanException.InvalidEmbeddingStoreFile(
                "Unexpected count: count=$existingCount"
            )
        }
        existingCount
    }

    override suspend fun writeHeader(channel: FileChannel, embeddingCount: Int) = withContext(Dispatchers.IO){
        val headerBuf = ByteBuffer.allocate(headerSize).order(ByteOrder.LITTLE_ENDIAN)
        headerBuf.putInt(embeddingCount)
        headerBuf.flip()
        channel.position(0)
        while (headerBuf.hasRemaining()) {
            channel.write(headerBuf)
        }
    }

}