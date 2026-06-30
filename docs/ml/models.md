# Models

## ModelLoader<T>

Generic abstraction for loading model assets.

Methods:

* `load(): T` — suspending load operation returning model data

---

## ModelAssetSource

Represents the origin of a model asset.

Variants:

* `LocalFile(file: File)`
* `Resource(resId: Int)`

---

## BaseModel<InputTensor, OutputTensor>

Abstract base class for inference models.

Fields:

* `loader: ModelLoader<*>` — model loader implementation

Methods:

* `loadModel()`
* `isLoaded(): Boolean`
* `run(inputs: Map<String, InputTensor>): Map<String, OutputTensor>`
* `close()`

Notes:

* Lifecycle-managed inference wrapper.
* Generic over both input and output tensor types.
* Implements `AutoCloseable`.

---

## OnnxModel

ONNX Runtime implementation of `BaseModel<OnnxTensor, OnnxValue>`.

Constructor:

```kotlin
OnnxModel(loader: ModelLoader<ByteArray>)
```

---

### loadModel()

Behavior:

* Loads model bytes using the configured `ModelLoader`.
* Creates an ONNX Runtime session.
* Executes on the IO dispatcher.

---

### isLoaded(): Boolean

Returns:

* `true` if an ONNX session has been created.
* `false` otherwise.

---

### run(inputs: Map<String, OnnxTensor>): Map<String, OnnxValue>

Executes ONNX inference.

Behavior:

* Requires an initialized session.
* Runs inference using the supplied `OnnxTensor` inputs.
* Returns a map of output names to `OnnxValue`s.
* Automatically closes all input tensors after execution.

Error handling:

* Throws `ModelNotInitialised` if the model has not been loaded.

---

### getInputNames(): List<String>?

Returns the model input node names if the session has been initialized.

---

### getOutputNames(): List<String>?

Returns the model output node names if the session has been initialized.

---

### getEnv(): OrtEnvironment

Returns the shared ONNX Runtime environment.

---

### close()

Behavior:

* Closes the ONNX Runtime session.
* Clears the current session reference.

---

## Design Notes

* Model loading is decoupled through `ModelLoader`.
* Supports file- and resource-based model loading.
* ONNX Runtime session creation is deferred until `loadModel()` is called.
* Uses native ONNX Runtime tensor types (`OnnxTensor` and `OnnxValue`).
* Automatically releases input tensor resources after inference.
* Designed for Android applications using ONNX Runtime.