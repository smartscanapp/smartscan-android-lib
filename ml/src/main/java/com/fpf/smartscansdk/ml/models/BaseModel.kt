package com.fpf.smartscansdk.ml.models

import com.fpf.smartscansdk.ml.models.loaders.ModelLoader


abstract class BaseModel<InputTensor> : AutoCloseable {
    protected abstract val loader: ModelLoader<*> // hidden implementation detail

    abstract suspend fun loadModel()
    abstract fun isLoaded(): Boolean
    abstract fun run(inputs: Map<String, InputTensor>): Map<String, Any>
}





