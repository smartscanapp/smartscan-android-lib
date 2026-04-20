package com.fpf.smartscansdk.ml.models

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(AndroidJUnit4::class)
class ModelManagerAndroidTest {

    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }

    @After
    fun teardown() {
        ModelRegistry.values.forEach { modelInfo ->
            ModelManager.deleteModel(context, modelInfo)
        }
    }

    @Test
    fun testGetModelFileReturnsCorrectPath() {
        ModelRegistry.forEach { (_, modelInfo) ->
            val file = ModelManager.getModelFile(context, modelInfo)
            assertTrue(file.absolutePath.contains(modelInfo.path))
        }
    }

    @Test
    fun testModelExistsReturnsFalseBeforeAnyAction() {
        ModelRegistry.keys.forEach { modelName ->
            assertFalse(ModelManager.modelExists(context, modelName))
        }
    }

    @Test
    fun testDeleteModelRemovesFilesOrDirectories() = runTest {
        ModelRegistry.values.forEach { modelInfo ->
            val targetFile = ModelManager.getModelFile(context, modelInfo)

            if (modelInfo.resourceFiles.isNullOrEmpty()) {
                // Single-file model
                targetFile.parentFile?.mkdirs()
                targetFile.writeText("dummy content")
            } else {
                // Directory model
                targetFile.mkdirs()
                modelInfo.resourceFiles.forEach { fileName ->
                    File(targetFile, fileName).writeText("dummy content")
                }
            }

            val deleted = ModelManager.deleteModel(context, modelInfo)
            assertTrue(deleted)
            assertFalse(targetFile.exists())
        }
    }

    @Test
    fun testListModelsListsOnlyExistingModels() = runTest {
        ModelRegistry.values.forEach { modelInfo ->
            val targetFile = ModelManager.getModelFile(context, modelInfo)

            if (modelInfo.resourceFiles.isNullOrEmpty()) {
                targetFile.parentFile?.mkdirs()
                targetFile.writeText("dummy content")
            } else {
                targetFile.mkdirs()
                modelInfo.resourceFiles.forEach { fileName ->
                    File(targetFile, fileName).writeText("dummy content")
                }
            }
        }

        val listedModels = ModelManager.listModels(context)
        ModelRegistry.keys.forEach { modelName ->
            assertTrue(listedModels.contains(modelName))
        }
    }

    @Test
    fun testDownloadModelInternalAndLoadProvidersForAllModels() = runTest {
        ModelRegistry.forEach { (modelName, modelInfo) ->

            val result = ModelManager.downloadModelInternal(context, modelInfo)

            if (result.isFailure) {
                val exception = result.exceptionOrNull()
                fail("Download failed for $modelName: ${exception?.message}")
            }

            assertTrue(ModelManager.modelExists(context, modelName))

            when (modelInfo.type) {
                ModelType.TEXT_ENCODER -> {
                    val provider = ModelManager.getTextEmbedder(context, modelName)
                    assertNotNull(provider)
                }
                ModelType.IMAGE_ENCODER -> {
                    val provider = ModelManager.getImageEmbedder(context, modelName)
                    assertNotNull(provider)
                }
                ModelType.OBJECT_DETECTOR -> {
                    val provider = ModelManager.getObjectDetector(context, modelName)
                    assertNotNull(provider)
                }
            }
        }
    }
}
