# Models

## ModelLoader<T>

Generic abstraction for loading model assets.

Methods:

* `load(): T` — suspending load operation returning model data

---

## FileOnnxLoader

Loads ONNX model from a local file.

Constructor:

```kotlin
FileOnnxLoader(file: File)
```

Behavior:

* Reads entire file into memory as `ByteArray`
* Intended for ONNX runtime session creation

---

## ResourceOnnxLoader

Loads ONNX model from Android raw resources.

Constructor:

```kotlin
ResourceOnnxLoader(resources: Resources, resId: Int)
```

Behavior:

* Opens raw resource stream
* Reads full content into `ByteArray`

---

## ModelAssetSource

Represents the origin of a model asset.

Variants:

* `LocalFile(file: File)`
* `Resource(resId: Int)`

---

## BaseModel<InputTensor>

Abstract base class for inference models.

Fields:

* `loader: ModelLoader<*>`

Methods:

* `loadModel()`
* `isLoaded(): Boolean`
* `run(inputs: Map<String, InputTensor>): Map<String, Any>`
* `close()`

Notes:

* Lifecycle-managed inference wrapper
* Input/output agnostic via generic tensor type

---

## TensorData

Unified tensor representation for ONNX inputs.

Variants:

### FloatBufferTensor

* `data: FloatBuffer`
* `shape: LongArray`

### IntBufferTensor

* `data: IntBuffer`
* `shape: LongArray`

### LongBufferTensor

* `data: LongBuffer`
* `shape: LongArray`

### DoubleBufferTensor

* `data: DoubleBuffer`
* `shape: LongArray`

### ShortBufferTensor

* `data: ShortBuffer`
* `shape: LongArray`
* `type: OnnxJavaType?`

### ByteBufferTensor

* `data: ByteBuffer`
* `shape: LongArray`
* `type: OnnxJavaType`

---

## OnnxModel

ONNX Runtime-based inference implementation of `BaseModel`.

Constructor:

```kotlin
OnnxModel(loader: ModelLoader<ByteArray>)
```

---

### loadModel()

Behavior:

* Loads model bytes via `ModelLoader`
* Creates ONNX Runtime session:

```kotlin
session = env.createSession(bytes)
```

* Runs on IO dispatcher

---

### isLoaded(): Boolean

Returns:

* `true` if session is initialized
* `false` otherwise

---

### run(inputs: Map<String, TensorData>): Map<String, Any>

Executes ONNX inference.

Behavior:

* Requires initialized session
* Converts `TensorData` → `OnnxTensor`
* Runs session inference
* Returns raw output map

Tensor creation:

```kotlin
OnnxTensor.createTensor(env, data, shape)
```

Special handling:

* Supports all primitive buffer types
* Optional ONNX type override for Short/Byte tensors

Cleanup:

* Closes all created tensors after execution

Error handling:

* Throws `ModelNotInitialised` if session is null

---

### getInputNames(): List<String>?

Returns model input node names if session is available.

---

### getEnv(): OrtEnvironment

Returns ONNX Runtime environment instance.

---

### close()

Behavior:

* Closes ONNX session
* Clears session reference

---

## Design Notes

* Model loading is decoupled via `ModelLoader`
* Supports file and resource-based model distribution
* ONNX session is created after explicit `loadModel`
* Tensor abstraction isolates ONNX APIs from upper layers
* Safe resource cleanup enforced during inference
* Designed for Android ONNX Runtime usage
