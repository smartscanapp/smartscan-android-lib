package com.fpf.smartscansdk.ml.providers.embeddings.clip

import android.content.Context
import android.util.JsonReader
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader

class ClipTokenizer(
    private val encoder: Map<String, Int>,
    private val bpeRanks: Map<Pair<String, String>, Int>,
) {
        companion object {

            fun load(context: Context, vocabResId: Int, mergesResId: Int): ClipTokenizer {
                val resources = context.resources
                val encoder = readEncoder(resources.openRawResource(vocabResId))
                val bpeRanks = readBpeRanks(resources.openRawResource(mergesResId))
                return ClipTokenizer(encoder, bpeRanks)
            }

            fun load(vocabFile: File, mergesFile: File): ClipTokenizer {
                val encoder = readEncoder(vocabFile.inputStream())
                val bpeRanks = readBpeRanks(mergesFile.inputStream())
                return ClipTokenizer(encoder, bpeRanks)
            }

            private fun readEncoder(stream: InputStream): Map<String, Int> {
                val map = hashMapOf<String, Int>()
                stream.use { s ->
                    val reader = JsonReader(InputStreamReader(s, "UTF-8"))
                    reader.beginObject()
                    while (reader.hasNext()) {
                        map[reader.nextName().replace("</w>", " ")] = reader.nextInt()
                    }
                    reader.close()
                }
                return map
            }

            private fun readBpeRanks(stream: InputStream): Map<Pair<String, String>, Int> {
                val map = hashMapOf<Pair<String, String>, Int>()
                stream.use { s ->
                    BufferedReader(InputStreamReader(s)).useLines { lines ->
                        lines.drop(1).forEachIndexed { i, line ->
                            val parts = line.split(" ")
                            map[parts[0] to parts[1].replace("</w>", " ")] = i
                        }
                    }
                }
                return map
            }
    }
    private val encodeRegex = Regex("""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""")

    fun encode(text: String): MutableList<Int> {
        val tokens = encodeRegex.findAll(text).map { result ->
            result.value.codePoints().boxed().map { byteEncoder[it]!! }.toArray().joinToString("")
        }
        return tokens.map { bpe(it) }.flatten().map { encoder[it]!! }.toMutableList()
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf("$token ")

        val wordWithBreak = token.map { it.toString() }.toMutableList()
        wordWithBreak[wordWithBreak.size - 1] = "${wordWithBreak[wordWithBreak.size - 1]} "
        var word = wordWithBreak.toList()
        var pairs = getPairs(word)

        while (true) {
            if (!pairs.any { bpeRanks.containsKey(it) }) break
            val (first, second) = pairs.minBy { bpeRanks.getOrDefault(it, Int.MAX_VALUE) }

            var i = 0
            val newWord = mutableListOf<String>()
            while (i < word.size) {
                val j = word.withIndex().indexOfFirst { it.index >= i && it.value == first }
                if (j != -1) {
                    newWord.addAll(word.subList(i, j))
                    i = j
                } else {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }

                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }

            word = newWord
            if (word.size == 1) {
                break
            } else {
                pairs = getPairs(word)
            }
        }
        return word
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return mutableSetOf<Pair<String, String>>().apply {
            for (i in 0 until word.size - 1) {
                add(word[i] to word[i + 1])
            }
        }
    }
}