package com.fpf.smartscansdk.core.models

import androidx.annotation.RawRes
import java.io.File

sealed interface ModelAssetSource {
    data class LocalFile(val file: File) : ModelAssetSource
    data class Resource(@RawRes val resId: Int) : ModelAssetSource
}