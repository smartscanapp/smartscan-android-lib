# Media Documentation

## ImageUtils

Utility functions for image preprocessing, cropping, scaling, face extraction, and visualization.

---

### centerCrop(bitmap, imageSize): Bitmap

Crops the input bitmap to a centered square and resizes it.

Behavior:

* If width ≥ height: crops horizontally centered region
* If height > width: crops vertically centered region
* Resizes result to `(imageSize, imageSize)`

Output:

* Square bitmap scaled to target size

---

### getScaledDimensions(width, height, maxSize): Pair<Int, Int>

Computes resized dimensions while preserving aspect ratio.

Behavior:

* If both dimensions ≤ maxSize: returns original size
* Otherwise scales largest dimension down to maxSize

Output:

* `(newWidth, newHeight)`

---

### getBitmapFromUri(context, uri, maxSize): Bitmap

Loads and decodes a bitmap from a content URI with size constraints.

Behavior:

* Uses `ImageDecoder`
* Applies `getScaledDimensions`
* Decodes bitmap at target resolution
* Returns mutable `ARGB_8888` copy

Output:

* Decoded and rescaled bitmap

---

### cropFaces(bitmap, boxes): List<Bitmap>

Extracts face regions from bounding boxes.

Behavior:

* Clamps box coordinates to bitmap bounds
* Skips invalid boxes
* Crops sub-bitmaps per bounding box

Output:

* List of cropped face bitmaps

---

### nms(boxes, scores, iouThreshold): List<Int>

Non-Maximum Suppression for filtering overlapping bounding boxes.

Behavior:

* Sorts boxes by confidence score
* Iteratively removes boxes with IoU above threshold
* Keeps highest scoring non-overlapping boxes

Output:

* Indices of selected boxes

---

### drawBoxes(bitmap, boxes, color, margin, strokeWidth): Bitmap

Draws bounding boxes onto a bitmap.

Behavior:

* Creates mutable bitmap copy
* Draws rectangles using `Canvas`
* Applies optional margin expansion
* Uses stroke-only rendering

Output:

* Annotated bitmap with drawn boxes

---

## VideoUtils

Utilities for extracting frames from video files.

---

### extractFramesFromVideo(context, videoUri, width, height, frameCount): List<Bitmap>?

Extracts evenly spaced frames from a video.

Behavior:

* Uses `MediaMetadataRetriever`
* Computes video duration
* Samples frames at uniform timestamps
* Scales frames to `(width, height)`
* Stops early if frame decoding fails
* Returns null on error or empty result

Execution:

* Runs on `Dispatchers.IO`

Output:

* List of extracted video frames or null

---

### Design Notes

* Frame extraction is time-based sampling (not scene detection)
* Designed for embedding pipelines requiring representative video frames
* Face utilities assume rectangular bounding boxes in pixel coordinates
* NMS implementation is linear after sorting (simple greedy suppression)
* All image outputs use mutable `ARGB_8888` format for downstream processing
