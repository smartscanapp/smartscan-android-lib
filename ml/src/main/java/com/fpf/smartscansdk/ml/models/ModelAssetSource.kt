package com.fpf.smartscansdk.ml.models

import android.content.res.Resources
import androidx.annotation.RawRes
import java.io.File

sealed interface ModelAssetSource {
    data class LocalFile(val file: File) : ModelAssetSource
    data class Resource(val resources: Resources, @RawRes val resId: Int) : ModelAssetSource
}