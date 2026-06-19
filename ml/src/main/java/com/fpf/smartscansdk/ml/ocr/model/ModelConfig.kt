package com.fpf.smartscansdk.ml.ocr.model

import android.content.Context
import com.fpf.smartscansdk.core.SmartScanException
import com.fpf.smartscansdk.ml.models.ModelAssetSource

internal data class ModelConfig(
    val characterList: List<String>,
) {
    companion object {
        fun parse(context: Context, assetSource: ModelAssetSource): ModelConfig {
            val content = try {
                val reader = when(assetSource){
                    is ModelAssetSource.Resource -> context.resources.openRawResource(assetSource.resId).bufferedReader()
                    is ModelAssetSource.LocalFile -> assetSource.file.bufferedReader()
                }
                reader.use { it.readText() }
            } catch (t: Throwable) {
                throw SmartScanException.ConfigParseFailed(cause = t)
            }
            val characterDict = try {
                extractCharacterDict(content)
            } catch (e: SmartScanException.ConfigParseFailed) {
                throw e
            } catch (t: Throwable) {
                throw SmartScanException.ConfigParseFailed( cause = t)
            }
            val charListWithSpace = characterDict.toMutableList().apply {
                if (lastOrNull() != " ") add(" ")
            }

            return ModelConfig(characterList = charListWithSpace)
        }

        private fun extractCharacterDict(content: String): List<String> {
            val lines = content.replace("\r\n", "\n").replace('\r', '\n').lines()
            val postProcessLine = lines.indexOfFirst { it.trim() == "PostProcess:" }
            if (postProcessLine < 0) {
                throw SmartScanException.ConfigParseFailed()
            }

            val postProcessIndent = leadingSpaces(lines[postProcessLine])
            val characterDictLine = findCharacterDictLine(lines, postProcessLine + 1, postProcessIndent)
            if (characterDictLine < 0) {
                throw SmartScanException.ConfigParseFailed()
            }

            val characterDictIndent = leadingSpaces(lines[characterDictLine])
            val characters = mutableListOf<String>()
            var lineIndex = characterDictLine + 1
            while (lineIndex < lines.size) {
                val line = lines[lineIndex]
                val indent = leadingSpaces(line)
                val content = line.substring(indent)
                val keyLikeContent = content.trim()
                if (keyLikeContent.isEmpty() || keyLikeContent.startsWith("#")) {
                    lineIndex++
                    continue
                }

                if (!content.startsWith("-")) {
                    if (indent <= characterDictIndent) break
                    lineIndex++
                    continue
                }

                characters.add(parseYamlListScalar(content.substring(1)))
                lineIndex++
            }

            if (characters.isEmpty()) {
                throw SmartScanException.ConfigParseFailed()
            }
            return characters
        }

        private fun findCharacterDictLine(
            lines: List<String>,
            startLine: Int,
            postProcessIndent: Int,
        ): Int {
            var lineIndex = startLine
            while (lineIndex < lines.size) {
                val line = lines[lineIndex]
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    lineIndex++
                    continue
                }
                val indent = leadingSpaces(line)
                if (indent <= postProcessIndent) break
                if (trimmed == "character_dict:") return lineIndex
                lineIndex++
            }
            return -1
        }

        private fun parseYamlListScalar(rawValue: String): String {
            val value = rawValue.dropWhile { it == ' ' }
            if (value.length >= 2 && value.first() == '\'' && value.last() == '\'') {
                return value.substring(1, value.length - 1).replace("''", "'")
            }
            if (value.length >= 2 && value.first() == '"' && value.last() == '"') {
                return value.substring(1, value.length - 1)
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n")
                    .replace("\\r", "\r")
                    .replace("\\t", "\t")
            }
            return value
        }

        private fun leadingSpaces(line: String): Int {
            return line.indexOfFirst { it != ' ' }.let { if (it < 0) line.length else it }
        }
    }
}
