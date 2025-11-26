package com.fpf.smartscansdk.ml.models.providers.embeddings

import com.fpf.smartscansdk.core.data.ClassificationError
import com.fpf.smartscansdk.core.data.ClassificationResult
import com.fpf.smartscansdk.core.data.PrototypeEmbedding
import com.fpf.smartscansdk.core.embeddings.dot
import com.fpf.smartscansdk.ml.providers.embeddings.classify
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

class FewShotClassifierTest {

    private val threshold = 0.4f
    private val minMargin = 0.05f

    private fun prototypeEmbedding(id: String, date: Long, values: FloatArray) =
        PrototypeEmbedding(id, date, values)

    @Test
    fun `classification fails when similarity below threshold`() = runTest {
        val embedding = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        val prototypes = listOf(
            prototypeEmbedding("1", 100, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)),
            prototypeEmbedding("2", 200, floatArrayOf(0.2f, 0.1f, 0.3f, 0.4f))
        )

        val result = classify(embedding, prototypes, threshold, minMargin)
        Assertions.assertTrue(result is ClassificationResult.Failure)
        Assertions.assertEquals(
            ClassificationError.THRESHOLD,
            (result as ClassificationResult.Failure).error
        )
    }

    @Test
    fun `classification fails when confidence margin below minimum`() = runTest {
        val embedding = floatArrayOf(1f, 1f, 1f, 1f)
        val prototypes = listOf(
            prototypeEmbedding("1", 100, floatArrayOf(1f, 1f, 1f, 1f)),
            prototypeEmbedding("2", 200, floatArrayOf(0.99f, 0.99f, 0.99f, 0.99f)),
        )

        val result = classify(embedding, prototypes, threshold, minMargin)
        Assertions.assertTrue(result is ClassificationResult.Failure)
        Assertions.assertEquals(
            ClassificationError.CONFIDENCE_MARGIN,
            (result as ClassificationResult.Failure).error
        )
    }

    @Test
    fun `classification succeeds when similarity and margin are sufficient`() = runTest {
        val embedding = floatArrayOf(1f, 0f, 0f, 0f)
        val prototypes = listOf(
            prototypeEmbedding("1", 100, floatArrayOf(1f, 0f, 0f, 0f)),
            prototypeEmbedding("2", 200, floatArrayOf(0f, 1f, 0f, 0f))
        )

        val result = classify(embedding, prototypes, threshold, minMargin)
        Assertions.assertTrue(result is ClassificationResult.Success)
        val success = result as ClassificationResult.Success
        Assertions.assertTrue(success.similarity >= threshold)
        Assertions.assertTrue(
            (success.similarity - (embedding dot prototypes[1].embeddings)) >= minMargin
        )
    }

    @Test
    fun `classification fails when less than 2 class prototypes are passed`() = runTest {
        val embedding = floatArrayOf(1f, 0f, 0f, 0f)
        val prototypes = listOf(
            prototypeEmbedding("1", 100, floatArrayOf(1f, 0f, 0f, 0f)),
        )

        val result = classify(embedding, prototypes, threshold, minMargin)
        Assertions.assertTrue(result is ClassificationResult.Failure)
        Assertions.assertTrue((result as ClassificationResult.Failure).error == ClassificationError.MINIMUM_CLASS_SIZE)
    }

}