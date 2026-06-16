package com.fpf.smartscansdk.ml.providers.ocr.util

internal object YamlUtils {

    fun leadingSpaces(line: String): Int {
        return line.indexOfFirst { it != ' ' }.let { if (it < 0) line.length else it }
    }
}
