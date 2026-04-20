package com.fpf.smartscansdk.ml.providers.embeddings.minilm

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

internal class MiniLmTokenizer(
    private val vocab: Map<String, Int>,
    private val doLowerCase: Boolean,
    private val unkToken: String,
    private val clsToken: String,
    private val sepToken: String,
) {

    companion object {

        private fun loadFromSources(vocabReader: BufferedReader, configReader: BufferedReader): MiniLmTokenizer {
            val vocabMap: Map<String, Int> = vocabReader.useLines { lines ->
                lines.mapIndexed { idx, token -> token to idx }.toMap()
            }

            val configText = configReader.use { it.readText() }
            val configJson = JSONObject(configText)

            return MiniLmTokenizer(
                vocab = vocabMap,
                doLowerCase = configJson.optBoolean("do_lower_case", true),
                unkToken = configJson.optString("unk_token", "[UNK]"),
                clsToken = configJson.optString("cls_token", "[CLS]"),
                sepToken = configJson.optString("sep_token", "[SEP]"),
            )
        }

        fun load(context: Context, vocabResId: Int, configResId: Int): MiniLmTokenizer {
            return loadFromSources(
                context.resources.openRawResource(vocabResId).bufferedReader(),
                InputStreamReader(context.resources.openRawResource(configResId), "UTF-8").buffered()
            )
        }
        fun load(vocabFile: File, configFile: File): MiniLmTokenizer {
            return loadFromSources(vocabFile.bufferedReader(), configFile.bufferedReader())
        }
    }

    private fun preprocess(text: String): String {
        var t = text
        if (doLowerCase) t = t.lowercase()
        t = t.replace(Regex("""([.,!?;:()"\'])"""), " $1 ")
        return t.trim().replace(Regex("\\s+"), " ")
    }

    private fun wordPiece(word: String): List<String> {
        val tokens = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var curToken: String? = null
            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) substr = "##$substr"
                if (vocab.containsKey(substr)) {
                    curToken = substr
                    break
                }
                end--
            }
            if (curToken == null) {
                tokens.add(unkToken)
                break
            } else {
                tokens.add(curToken)
                start = end
            }
        }
        return tokens
    }

    fun tokenize(text: String): List<String> {
        val pre = preprocess(text)
        return pre.split(" ").flatMap { wordPiece(it) }
    }

    fun encode(text: String): Pair<IntArray, IntArray> {
        val tokens = listOf(clsToken) + tokenize(text) + listOf(sepToken)
        val ids = IntArray(tokens.size)
        val mask = IntArray(tokens.size)

        for (i in tokens.indices) {
            ids[i] = vocab[tokens[i]] ?: vocab[unkToken]!!
            mask[i] = 1
        }
        return ids to mask
    }
}
