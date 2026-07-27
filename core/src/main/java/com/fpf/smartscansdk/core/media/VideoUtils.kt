package com.fpf.smartscansdk.core.media

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun extractFramesFromVideo(context: Context, videoUri: Uri, width: Int, height: Int,frameCount: Int = 10): List<Bitmap>? = withContext(Dispatchers.IO) {
    val retriever = MediaMetadataRetriever()
    val extractor = MediaExtractor()

    return@withContext try {
        retriever.setDataSource(context, videoUri)
        extractor.setDataSource(context, videoUri, null)

        val frameList = mutableListOf<Bitmap>()
        val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLong()?.times(1000) ?: return@withContext null

        val codec = (0 until extractor.trackCount)
            .map { extractor.getTrackFormat(it) }
            .firstOrNull { it.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true }
            ?.getString(MediaFormat.KEY_MIME)

        for (i in 0 until frameCount) {
            val frameTimeUs = (i * durationUs) / frameCount
            val bitmap = retriever.getScaledFrameAtTime(
                frameTimeUs,
                MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                width,
                height
            )?: throw SmartScanException.UnsupportedVideoCodec( "Video codec or profile not supported: $codec")
            frameList.add(bitmap)
        }

        if (frameList.isEmpty()) return@withContext  null

        frameList
    } catch (e: Exception) {
        Log.e("extractFramesFromVideo", "Error extracting frames", e)
        null
    } finally {
        retriever.release()
        extractor.release()
    }
}