## v1.2.1 - 17/12/2025

### Added
* Added support for filterIds when querying

### Fixed

* Fixed duplication bug in `add` method of `FileEmbeddingStore`
* Fixed get(ids) return no results due to empty cache

### Changed
* Queries now return List<Long>
* Indexer call `onBatchComplete` on listener if provided

### Removed
* Removed `useCache`


## v1.2.0 - 26/11/2025

### Added
* Added new models: DinoV2 and InceptionResnet

### Removed
* Removed `FileEmbeddingRetriever` and and its query methods to `FileEmbeddingStore` (breaking)
* Removed `iscached` from `FileEmbeddingStore` (breaking)
* Removed `ProcessOptions` (breaking)
* Removed data packages for both core and ml modules and moved the files into relevant packages

## v1.1.1 - 04/11/2025

### Added
* Added new text embedding provider, Mini-LM
*  Add `initialized` and `isInitialized` to `IEmbeddingProvider`"


### Changed
* IEmbeddingProvider is require to provider `embeddingDim` variable (used to be optional)
* Renamed `embeddingLength` to `embeddingDim` for `FileEmbeddingStore` constructor param
* Move interfaces:
  - Moved to core/embeddings: `IEmbeddingStore`, `IRetriever`, `IEmbeddingProvider`
  - Moved to core/processor: `IProcessorListener`
  - Moved to ml/models: `IModelLoader`

## v1.1.0 - 30/10/2025

### Changed
* Project structure refactored from **core + extensions** to **core + ml**.
* Imports updated accordingly:
  - **core** → minimal runtime: shared interfaces, data classes, embeddings, media helpers, processor execution, and efficient batch/concurrent processing.
  - **ml** → ML infrastructure and models: model loaders, base models, embedding providers (e.g., CLIP), and few-shot classifiers. Optional or experimental ML-related features can be added under `ml/providers`.
  - Both modules organize contracts and data classes under their own `data/` packages.

* All `IEmbeddingProviders` must now implement `embedBatch`
* `ClipImageEmbedder` and `ClipTextEmbedder` now accept context instead of resources
* `BatchProcessor` now accepts a `Context` (uses `applicationContext` internally).

### Fixed

* fix `ClipTextEmbedder`: prevent IllegalCapacity in embed


### Removed

* `Organiser` class removed.

### Notes
This release replaces the old `core` and `extensions` structure.  
If you are upgrading from ≤1.0.4, update imports and Gradle dependencies.


## v1.0.4 – 19/10/2025

### Changed

* Pass file directly in `FileEmbeddingStore` constructor instead of dir and filename
* Update batch processor to ensure progress is tracked correctly regardless of errors
* Update batch processor to call onComplete even if items is empty

## v1.0.3 – 14/10/2025

### Added

* `FileEmbeddingRetriever` now supports batch retrieval via `start` and `end` indices with a new `query` overload.
* `FileEmbeddingStore` `getAll` method renamed to `get`, and two new overloads added:
  * `get(ids: List<Long>)` – fetch multiple embeddings by ID.
  * `get(id: Long)` – fetch a single embedding by ID.
* Tests added to verify correct behavior and boundary handling for the new query overload.

## v1.0.2 – 05/10/2025

### Changed
* Moved MemoryUtils into processor
* Moved IProcessorListener to its own file
* Moved MemoryOptions into ProcessorData.kt
* Update indexers to users correctly named parameter item instead of id to prevent issues with named parameters

### Fixed
* Fixed typo in getScaledDimension function

## v1.0.1 – 26/09/2025

### Changed
* IEmbeddingStore interface - getAll, isCached, exist
* Use linked hashmap for cache instead of list
* Pass store to Indexers
* Update tests

## v1.0.0 – 23/09/2025
* Initial release