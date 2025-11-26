package com.fpf.smartscansdk.ml.models.loaders

import android.content.res.Resources
import androidx.annotation.RawRes
import java.io.File

class FileOnnxLoader(private val path: String) : IModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = File(path).readBytes()
}

class ResourceOnnxLoader(private val resources: Resources, @RawRes private val resId: Int) :
    IModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = resources.openRawResource(resId).readBytes()
}
