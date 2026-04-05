package com.fpf.smartscansdk.ml.models.loaders

import android.content.res.Resources
import androidx.annotation.RawRes
import java.io.File

class FileOnnxLoader(private val file: File) : IModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = file.readBytes()
}

class ResourceOnnxLoader(private val resources: Resources, @RawRes private val resId: Int) :
    IModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = resources.openRawResource(resId).readBytes()
}
