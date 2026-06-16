// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.fpf.smartscansdk.ml.providers.ocr.model

import com.fpf.smartscansdk.ml.models.ModelAssetSource

sealed class OCRError(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class ModelNotFound(assetSource: ModelAssetSource, cause: Throwable? = null) : OCRError("Model not found: $assetSource", cause)
    class ModelLoadFailed(modelName: String, cause: Throwable) : OCRError("Failed to load $modelName model", cause)
    class ConfigParseFailed(cause: Throwable? = null) : OCRError("Failed to parse config", cause)
    class InvalidImage : OCRError("Input image is empty or invalid")
    class InferenceFailed(stage: String, cause: Throwable) : OCRError("Inference failed at stage '$stage'", cause)
    class DecodeError(message: String, cause: Throwable? = null) : OCRError(message, cause)
}
