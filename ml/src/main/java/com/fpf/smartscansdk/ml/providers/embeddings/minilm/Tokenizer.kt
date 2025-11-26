package com.fpf.smartscansdk.ml.providers.embeddings.minilm

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.InputStreamReader
import kotlin.collections.toLongArray

class MiniLmTokenizer(
    private val vocab: Map<String, Int>,
    private val maxLen: Int,
    private val doLowerCase: Boolean,
    private val unkToken: String,
    private val clsToken: String,
    private val sepToken: String,
    private val padToken: String
) {

    companion object {
         fun fromRawResources(context: Context, vocabResId: Int, configResId: Int): MiniLmTokenizer {
                // Load vocab
                val vocabMap: Map<String, Int> = context.resources.openRawResource(vocabResId)
                    .bufferedReader()
                    .useLines { lines -> lines.mapIndexed { idx, token -> token to idx }.toMap() }


                // Load config
                val configText = context.resources.openRawResource(configResId).use { input ->
                    InputStreamReader(input, "UTF-8").readText()
                }
                val configJson = JSONObject(configText)

                return MiniLmTokenizer(
                    vocab = vocabMap,
                    maxLen = configJson.optInt("max_length", 128),
                    doLowerCase = configJson.optBoolean("do_lower_case", true),
                    unkToken = configJson.optString("unk_token", "[UNK]"),
                    clsToken = configJson.optString("cls_token", "[CLS]"),
                    sepToken = configJson.optString("sep_token", "[SEP]"),
                    padToken = configJson.optString("pad_token", "[PAD]")
                )
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

    fun encode(text: String): Pair<LongArray, LongArray> {
        val tokens = listOf(clsToken) + tokenize(text) + listOf(sepToken)
        val ids = tokens.map { vocab[it] ?: vocab[unkToken]!! }.toMutableList()
        val attention = MutableList(ids.size) { 1L }

        if (ids.size > maxLen) {
            ids.subList(maxLen, ids.size).clear()
            attention.subList(maxLen, attention.size).clear()
        } else if (ids.size < maxLen) {
            val padLength = maxLen - ids.size
            ids.addAll(List(padLength) { vocab[padToken]!! })
            attention.addAll(List(padLength) { 0L })
        }

        return ids.map { it.toLong() }.toLongArray() to attention.map { it }.toLongArray()
    }
}
