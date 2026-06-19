package com.fpf.smartscansdk.ml.models.loaders

import android.content.res.Resources
import androidx.annotation.RawRes
import java.io.File

class FileLoader(private val file: File) : ModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = file.readBytes()
}

class ResourceLoader(private val resources: Resources, @RawRes private val resId: Int) :
    ModelLoader<ByteArray> {
    override suspend fun load(): ByteArray = resources.openRawResource(resId).readBytes()
}
