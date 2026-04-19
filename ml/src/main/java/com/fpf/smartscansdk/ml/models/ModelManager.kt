package com.fpf.smartscansdk.ml.models

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.net.toUri
import com.fpf.smartscansdk.core.SmartScanException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream
import kotlin.collections.forEach

object ModelManager {
    const val ROOT_DIR = "models"

    suspend fun downloadModelInternal(context: Context, modelInfo: ModelInfo, onProgress: ((progress: Int) -> Unit)? = null): Result<File> =
        withContext(Dispatchers.IO) {
            val targetFileOrDir = getModelFile(context, modelInfo)
            val isZipTarget = !modelInfo.resourceFiles.isNullOrEmpty()
            var tempZipFile: File? = null

            try {
                targetFileOrDir.parentFile?.mkdirs()

                if (isZipTarget) {
                    tempZipFile = File(targetFileOrDir.absolutePath + ".zip")
                    val downloadResult =
                        downloadFileInternal(modelInfo.url, tempZipFile, onProgress)
                    if (downloadResult.isFailure) return@withContext Result.failure(downloadResult.exceptionOrNull()!!)

                    if (!targetFileOrDir.exists()) targetFileOrDir.mkdirs()

                    val zipFile = downloadResult.getOrThrow()
                    val extractedFiles = unzipFiles(zipFile, targetFileOrDir)
                    val extractedPaths = extractedFiles.map { it.path }
                    val isValid = extractedPaths.all { extractedPath ->
                        modelInfo.resourceFiles.any { dependency ->
                            extractedPath.contains(
                                dependency
                            )
                        }
                    }

                    if (!isValid) {
                        extractedFiles.forEach { it.delete() }
                        return@withContext Result.failure(SmartScanException.InvalidModelFile())
                    }

                    Result.success(targetFileOrDir)
                } else {
                    downloadFileInternal(modelInfo.url, targetFileOrDir, onProgress)
                }
            } catch (e: Exception) {
                Result.failure(e)
            } finally {
                tempZipFile?.delete()
            }
        }

    fun downloadModelExternal(context: Context, url: String){
        val intent = Intent(Intent.ACTION_VIEW, url.toUri())
        context.startActivity(intent)
    }

    suspend fun importModel(context: Context, modelInfo: ModelInfo, uri: Uri) =
        withContext(Dispatchers.IO) {
            val targetFileOrDir = getModelFile(context, modelInfo)
            val isZipTarget = !modelInfo.resourceFiles.isNullOrEmpty()

            if (isZipTarget) {
                val tempFile = File.createTempFile("model_", ".zip", context.cacheDir)
                tempFile.parentFile?.mkdirs()

                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    FileOutputStream(tempFile).use { outputStream -> inputStream.copyTo(outputStream) }
                }

                if (!targetFileOrDir.exists()) targetFileOrDir.mkdirs()

                val extractedFiles = unzipFiles(tempFile, targetFileOrDir)
                val extractedFilesPaths = extractedFiles.map { it.path }
                val isValid = extractedFilesPaths.all { extractedPath ->
                    modelInfo.resourceFiles.any { dependency -> extractedPath.contains(dependency) }
                }

                try {
                    if (!isValid) {
                        extractedFiles.forEach { it.delete() }
                        throw SmartScanException.InvalidModelFile()
                    }
                } finally {
                    tempFile.delete()
                }
            } else {
                targetFileOrDir.parentFile?.mkdirs()
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    FileOutputStream(targetFileOrDir).use { outputStream ->
                        inputStream.copyTo(
                            outputStream
                        )
                    }
                }
            }
        }

    fun modelExists(context: Context, model: ModelName): Boolean{
        val modelInfo = ModelRegistry[model]!!
        val file = getModelFile(context, modelInfo)

        if(file.isFile) return file.exists()

        if (file.isDirectory){
            if (!file.exists()) return false
            return modelInfo.resourceFiles?.all{
                val resourceFile = File(file, it)
                return resourceFile.exists()
            }?: false
        }

        return false
    }

    fun deleteModel(context: Context, modelInfo: ModelInfo): Boolean{
        val file = getModelFile(context, modelInfo)
        return if(file.isDirectory){
            file.deleteRecursively()
        }else{
            file.delete()
        }
    }

    fun listModels(context: Context, type: ModelType? = null): List<ModelName> {
        return ModelRegistry.filter { entry ->
            modelExists(context, entry.key) && (type == null || entry.value.type == type)
        }.keys.toList()
    }

    fun getModelFile(context: Context, modelInfo: ModelInfo): File {
        return File(File(context.filesDir, ROOT_DIR), modelInfo.path)
    }

    private suspend fun unzipFiles(zipFile: File, targetDir: File): List<File> =
        withContext(Dispatchers.IO) {
            if (!targetDir.exists()) targetDir.mkdirs()

            val extractedFiles = mutableListOf<File>()

            ZipInputStream(FileInputStream(zipFile)).use { zipInputStream ->
                var entry = zipInputStream.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory) {
                        val entryFile = File(targetDir, File(entry.name).name)
                        FileOutputStream(entryFile).use { outputStream ->
                            zipInputStream.copyTo(outputStream)
                        }
                        extractedFiles.add(entryFile)
                    }
                    zipInputStream.closeEntry()
                    entry = zipInputStream.nextEntry
                }
            }

            extractedFiles
        }

    private suspend fun downloadFileInternal(url: String, outputFile: File, onProgress: ((progress: Int) -> Unit)? = null): Result<File> =
        withContext(Dispatchers.IO) {
            val tempFile = File(outputFile.absolutePath + ".tmp")
            var lastProgress = 0

            // Cleanup any leftover temp file from previous crashes/interruption
            if (tempFile.exists()) {
                tempFile.delete()
            }

            try {
                val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 15_000
                    readTimeout = 15_000
                    doInput = true
                    connect()
                }

                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    return@withContext Result.failure(
                        SmartScanException.ModelDownloadFailed("HTTP error code: ${connection.responseCode}")
                    )
                }

                val contentLength = connection.contentLengthLong

                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(8 * 1024)
                        var bytesRead: Int
                        var totalBytesRead = 0L

                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead

                            if (contentLength > 0 && onProgress != null) {
                                val progress = (totalBytesRead * 100 / contentLength).toInt()
                                if (progress > lastProgress) {
                                    withContext(Dispatchers.Main) {
                                        onProgress(progress)
                                        lastProgress = progress
                                    }
                                }
                            }
                        }

                        output.flush()
                    }
                }

                // Replace existing file safely
                if (outputFile.exists()) {
                    outputFile.delete()
                }

                if (!tempFile.renameTo(outputFile)) {
                    tempFile.delete()
                    return@withContext Result.failure(SmartScanException.ModelDownloadFailed("Failed to rename temp model file"))
                }

                Result.success(outputFile)

            } catch (e: Exception) {
                tempFile.delete()
                Result.failure(e)
            }
        }
}