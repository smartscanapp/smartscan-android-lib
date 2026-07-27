package com.fpf.smartscansdk.core.file

import android.util.Log
import io.mockk.every
import io.mockk.mockkStatic
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class TombStoneStoreTest {

    @BeforeEach
    fun setup() {
        mockkStatic(Log::class)
        every { Log.d(any<String>(), any<String>()) } returns 0
        every { Log.e(any<String>(), any<String>()) } returns 0
        every { Log.i(any<String>(), any<String>()) } returns 0
        every { Log.w(any<String>(), any<String>()) } returns 0
    }

    @TempDir
    lateinit var tempDir: File

    private val tombStoneFile: File
        get() = File(tempDir, "test.tombstone")

    private val tombStone: TombstoneStore
        get() = TombstoneStore(tombStoneFile)

    private fun genIds(n: Int): List<Long> = List(n){i -> i.toLong()}


    @Test
    fun `tombstone appends and read always returns unique ids`() = runTest {
        val ids = genIds(10)
        tombStone.append(ids)
        val tombStonedIds = tombStone.read()

        Assertions.assertEquals(ids.toSet(), tombStonedIds)

        tombStone.append(ids)
        val tombStonedIdsBatch2 = tombStone.read()
        Assertions.assertEquals(ids.toSet(), tombStonedIdsBatch2)
    }

    @Test
    fun `clear removes tombStoneFile`() = runTest {
        val ids = genIds(10)
        tombStone.append(ids)
        assertTrue(tombStone.exists)

        tombStone.clear()
        assertFalse(tombStone.exists)
    }

    @Test
    fun `recover item from the tombstone`() = runTest {
        val ids = genIds(10)
        tombStone.append(ids)
        assertTrue(tombStone.exists)
        val idRecover = ids.first()
        tombStone.recoverIfNeeded(listOf(idRecover))

        assertTrue(idRecover !in tombStone.read())
    }
}