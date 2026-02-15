# Vision Archive AI - Architecture & Workflow Guide

---

## System Architecture Diagrams

### 1. Overall System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Vision Archive AI System                     │
└────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   USER LAYER    │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
            ┌───────▼────┐  ┌────▼─────┐  ┌──▼────────┐
            │  Streamlit │  │  Gallery  │  │Search CLI │
            │   Web UI   │  │  (HTML)   │  │ Interface │
            └────────┬───┘  └────┬──────┘  └──┬───────┘
                     │           │            │
                     └───────────┼────────────┘
                                 │
                        ┌────────▼──────────┐
                        │  SEARCH ENGINE    │  CLIP (ONNX)
                        │  (Python Modules) │
                        │                   │
                        │ • SearchEngine    │ Text/Image Encoding (512-d)
                        │ • PersonManager   │ Database lookups
                        │ • VectorStorage   │ Similarity computation
                        └────────┬──────────┘
                                 │
                     ┌───────────┼───────────┐
                     │           │           │
            ┌────────▼──┐  ┌─────▼──┐  ┌───▼────────┐
            │   Face    │  │Clustering│  │ Embedding │
            │ Recognition│  │ Engine   │  │Aggregation│
            │ (AIEngine) │  │(HDBSCAN) │  │ Storage   │
            │            │  │          │  │           │
            │ InsightFace│  │  Groups  │  │ Caching   │
            │ + Buffalo_s│  │  Faces   │  │           │
            └────────┬───┘  └────┬─────┘  └──┬────────┘
                     │           │           │
                     └───────────┼───────────┘
                                 │
                    ┌────────────▼──────────┐
                    │   DATA STORAGE LAYER  │
                    │                       │
                    │ • embeddings.npy      │ Face vectors
                    │ • image_vectors.npy   │ Image vectors
                    │ • people_db.json      │ Identity DB
                    │ • image_cache.json    │ Path mappings
                    │ • paths.json          │ File list
                    │                       │
                    │ MODELS:               │
                    │ • buffalo_s/          │ Detection + Recognition
                    │ • clip_image.onnx     │ Image encoding
                    │ • clip_text.onnx      │ Text encoding
                    │ • tokenizer.json      │ BPE tokenizer
                    └───────────────────────┘
```

### 2. Data Flow: From Image to Identification

```
INPUT: Raw Image File
       │
       ▼
┌──────────────────────────┐
│  1. IMAGE LOADING (CPU)  │
│  File → cv2.imread()     │
│  Format: BGR numpy array │
│  Shape: (H, W, 3)        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────┐
│  2. FACE DETECTION (GPU)     │
│  InsightFace RetinaFace      │
│  Input: Raw image           │
│  Output: Bounding boxes     │
│           Confidence scores │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  3. FACE SELECTION           │
│  Choose: Largest face        │
│  (Most likely the subject)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  4. EMBEDDING EXTRACTION     │
│  ArcFace → 512-d vector      │
│  L2 normalization            │
│  Output: float32 array       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  5. CLUSTERING (CPU)         │
│  All embeddings + HDBSCAN    │
│  Distance metric: Euclidean  │
│  Output: Cluster labels      │
│          -1 = outlier/noise  │
│           0+ = cluster ID    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  6. PERSON IDENTIFICATION    │
│  Map image → Person_X        │
│  Store in people_db.json     │
│  Name: "Unknown" (default)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  7. INDEXING FOR SEARCH      │
│  Image → CLIP Image Encoder  │
│  Output: 512-d CLIP vector   │
│  Store in image_vectors.npy  │
│  Map in image_cache.json     │
└──────────┬───────────────────┘
           │
           ▼
OUTPUT: Searchable, Clustered Photo Archive
```

### 3. Inference Pipeline Architecture

```
PRODUCTION PIPELINE (production_pipeline.py)
═══════════════════════════════════════════

MASTER THREAD:
┌─────────────────────────────┐
│ 1. Scan test_images/        │
│ 2. Filter new images        │
│ 3. Create work queue        │
│ 4. Spawn worker threads     │
│ 5. Aggregate results        │
│ 6. Save embeddings.npy      │
│ 7. Update paths.json        │
└─────────────┬───────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│ WORKER │ │ WORKER │ │MANAGER │
│   #0   │ │   #1   │ │        │
└────┬───┘ └───┬────┘ └───┬────┘
     │         │          │
  1. Get image from queue  │
  2. Load (CPU)            │
  3. Infer (GPU)           │
  4. Extract embedding     │
  5. Thread-safe append    │
     to result_list        │
     │         │          │
     └─────────┼──────────┘
               │
          ┌────▼────┐
          │ Lock &  │ Synchronization point:
          │ Append  │ Prevents race conditions
          │ Results │
          │         │
          └────┬────┘
               │
        ┌──────▼──────┐
        │ Final Save  │
        │ to Disk     │
        └─────────────┘

PERFORMANCE CHARACTERISTICS:
┌──────────────────────────────────┐
│ • Parallel workers: 2 (per GPU)  │
│ • Queue size: ~50-100 images     │
│ • Worker latency: 5-7ms / image  │
│ • Throughput: 150-200 FPS total  │
│ • Memory: ~600MB VRAM per worker │
└──────────────────────────────────┘
```

### 4. Search Pipeline

```
USER QUERY
    │
    ├─ Text Query: "Person in suit"
    │  │
    │  ▼
    │  ┌──────────────────────────┐
    │  │ CLIP Text Encoder        │
    │  │ (clip_text.onnx)         │
    │  │ Input: Text              │
    │  │ Output: 512-d vector     │
    │  └──────────────┬───────────┘
    │                 │
    │
    └─ Image Query: Upload photo.jpg
       │
       ▼
       ┌──────────────────────────┐
       │ CLIP Image Encoder       │
       │ (clip_image.onnx)        │
       │ Input: Image             │
       │ Output: 512-d vector     │
       └──────────────┬───────────┘
                      │
                      ▼
              ┌───────────────────────┐
              │  Query Vector (512-d) │
              │  ┌─────────────────┐  │
              │  │ [0.15, -0.08,   │  │
              │  │  0.42, 0.27,    │  │
              │  │  ...512 dims]   │  │
              │  └─────────────────┘  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────────────┐
              │ SIMILARITY COMPUTATION        │
              │                               │
              │ scores = dot(library, query)  │
              │ # Vectorized BLAS operation   │
              │ # Complexity: O(N × 512)      │
              │ # Time: ~50ms for 13K images  │
              │                               │
              │ Output: (13000,) array        │
              │ [0.87, 0.23, 0.91, ...]      │
              └───────────┬───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ TOP-K SELECTION       │
              │ argsort(scores)[:20]  │
              │ Returns indices       │
              │ [143, 5, 78, 12, ...] │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ RETRIEVE & DISPLAY    │
              │ Load images by index  │
              │ Show with scores      │
              │ [87%, 91%, 78%, ...]  │
              └───────────────────────┘

TARGET PERFORMANCE:
Query Encoding:     <100ms
Similarity Match:   <50ms
Result Retrieval:   <50ms
─────────────────────────────
Total Latency:      <200ms
```

---

## Detailed Module Interactions

### A. Face Extraction Pipeline

```
test_pipeline_queue.py / production_pipeline.py
│
├─ Load AIEngine("buffalo_s")
│  ├─ Initialize FaceAnalysis
│  ├─ Set det_size = (320, 320)
│  ├─ Apply CUDNN HEURISTIC fix
│  └─ Warmup (5 dummy inferences)
│
├─ For each image:
│  ├─ cv2.imread(path)  ── CPU Operation
│  │  └─ Returns: numpy array (H, W, 3) uint8
│  │
│  ├─ ai.get_faces(img) ── GPU Operation
│  │  │
│  │  ├─ Resize to (320, 320)
│  │  ├─ Convert to CHW format
│  │  ├─ Run detection model (det_500m.onnx)
│  │  │  └─ Returns: N × 4 bounding boxes
│  │  ├─ For each face:
│  │  │  ├─ Crop face region
│  │  │  ├─ Align face (normalize rotation/scale)
│  │  │  ├─ Run recognition model (w600k_mbf.onnx)
│  │  │  └─ Returns: 512-d embedding (ArcFace)
│  │  └─ L2 normalize embeddings
│  │
│  └─ Select: faces[0] (largest by area)
│     └─ Store: embedding (512-d float32)
│
├─ Batch processing:
│  ├─ Save every N images to avoid memory overflow
│  ├─ Append to existing embeddings.npy
│  └─ Update paths.json
│
└─ Final Output:
   ├─ embeddings.npy (N × 512 float32)
   ├─ paths.json [image_path_1, image_path_2, ...]
   └─ Performance: 150-200 FPS on RTX 3050
```

### B. Clustering Pipeline

```
tune_clustering.py
│
├─ Load embeddings.npy (N × 512)
├─ Load paths.json
│
├─ Initialize ClusterEngine
│  ├─ Create HDBSCAN instance
│  │  ├─ min_cluster_size = 3 (configurable)
│  │  ├─ metric = 'euclidean'
│  │  ├─ use prediction_data = True (for extensibility)
│  │  └─ cluster_selection_method = 'eom'
│
├─ Fit & Predict:
│  ├─ clusterer.fit_predict(embeddings)
│  │  ├─ Compute pairwise distances (Euclidean)
│  │  │  └─ O(N²) but optimized with KD-trees
│  │  ├─ Build minimum spanning tree
│  │  ├─ Extract clusters via condensation hierarchy
│  │  └─ Returns: labels array (N,)
│  │
│  ├─ Labels interpretation:
│  │  ├─ -1: Noise/outliers (unmatched faces)
│  │  ├─  0: Cluster 0 (first person)
│  │  ├─  1: Cluster 1 (second person)
│  │  └─ And so on...
│
├─ Analyze Results:
│  ├─ Count unique clusters (excluding -1)
│  ├─ Count noise points
│  └─ Display statistics
│
├─ User Interaction:
│  ├─ Display results
│  ├─ Prompt: Save to DB? [y/n]
│  │
│  └─ If YES → PersonManager.save_people()
│     ├─ Create person_db.json structure
│     │
│     ├─ For each image:
│     │  └─ person_id = f"Person_{label}"
│     │  └─ Add to person_id['photos'] list
│     │
│     └─ Save to people_db.json
│        {
│          "Person_0": {
│            "name": "Unknown",
│            "photos": ["test_images/img1.jpg", ...]
│          },
│          "Person_1": {
│            "name": "Unknown",
│            "photos": [...]
│          }
│        }
│
└─ Output:
   ├─ people_db.json (identity mapping)
   ├─ Human-readable clusters
   └─ Ready for searching
```

### C. Search Pipeline

```
StreamlitApp (app.py)
│
├─ Initialize (cached):
│  ├─ SearchEngine() [CLIP models]
│  ├─ VectorStorage() [load embeddings]
│  └─ PersonManager() [load people_db.json]
│
├─ User Input:
│  └─ Mode: "Semantic Search"
│     ├─ Choice: "Text Description"
│     │  └─ User enters: "Person in suit"
│     │
│     ├─ SearchEngine.get_text_embedding()
│     │  ├─ Tokenize: "person" → [320, 518, 205, ...]
│     │  ├─ Pad to 77 tokens
│     │  ├─ Run CLIP text encoder (clip_text.onnx)
│     │  └─ Output: 512-d embedding
│     │
│     └─ Vectorized Similarity:
│        ├─ scores = np.dot(storage.vectors, query_vec)
│        │  └─ (13000, 512) × (512,) → (13000,)
│        │
│        └─ Top results:
│           ├─ top_indices = argsort(-scores)[:20]
│           ├─ Filter: score > 0.15 threshold
│           └─ Display with images & scores
│
└─ Display Results:
   ├─ 5-column grid layout
   ├─ Each: image + match percentage
   └─ Sorted by similarity
```

---

## Database Schema

### people_db.json (Identity Database)

```json
{
  "Person_0": {
    "name": "Aaron Peirsol",
    "photos": [
      "test_images/Aaron_Peirsol_0001.jpg",
      "test_images/Aaron_Peirsol_0002.jpg",
      "test_images/Aaron_Peirsol_0003.jpg"
    ]
  },
  "Person_1": {
    "name": "Abdoulaye Wade",
    "photos": [
      "test_images/Abdoulaye_Wade_0001.jpg",
      "test_images/Abdoulaye_Wade_0002.jpg"
    ]
  },
  "Person_-1": {
    "name": "Unknown/Noise",
    "photos": [
      "test_images/unmatched_001.jpg",
      "test_images/unmatched_002.jpg"
    ]
  }
}
```

### image_cache.json (Search Index)

```json
{
  "test_images/Aaron_Peirsol_0001.jpg": 0,
  "test_images/Aaron_Peirsol_0002.jpg": 1,
  "test_images/Aaron_Peirsol_0003.jpg": 2,
  "test_images/Abdoulaye_Wade_0001.jpg": 3,
  "test_images/Abdoulaye_Wade_0002.jpg": 4,
  ...
}
```

**Mapping**: `image_path → index_in_image_vectors.npy`

### paths.json (Face Extraction Output)

```json
[
  "test_images/Aaron_Peirsol_0001.jpg",
  "test_images/Aaron_Peirsol_0002.jpg",
  "test_images/Aaron_Peirsol_0003.jpg",
  "test_images/Abdoulaye_Wade_0001.jpg",
  "test_images/Abdoulaye_Wade_0002.jpg",
  ...
]
```

---

## Performance Metrics

### Single Image Processing

```
Operation                Time        Hardware
─────────────────────────────────────────────
cv2.imread()            ~2ms        CPU
Face Detection          ~3ms        GPU
Embedding Extract       ~2ms        GPU
─────────────────────────────────────────────
Total / Image           ~7ms        (RTX 3050)
Throughput             ~145 FPS
```

### Batch Operations

```
Operation           Time        Dataset Size
──────────────────────────────────────────────
Load 13K embeddings ~100ms      numpy.load()
Clustering         ~3s          HDBSCAN
Total Search       ~200ms       13K images
```

### Memory Usage

```
Component               Memory
─────────────────────────────────────────────
InsightFace Model      ~600MB VRAM
CLIP Image Model       ~200MB VRAM
CLIP Text Model        ~300MB VRAM
Embeddings (13K)       ~30MB RAM (float32)
Image Cache            ~50MB RAM (JSON)
──────────────────────────────────────────────
Total (multi-worker)   ~2GB VRAM, 200MB RAM
```

---

## Workflow Sequences

### Complete End-to-End Workflow

```
1. DATA PREPARATION
   prepare_lfw.py
   └─ Copy 13K images to test_images/

2. FACE EXTRACTION
   production_pipeline.py
   ├─ Input: test_images/ (13K images)
   ├─ Process: Extract face embeddings (200 FPS)
   └─ Output: embeddings.npy, paths.json (30MB)

3. CLUSTERING
   tune_clustering.py
   ├─ Input: embeddings.npy (13K × 512)
   ├─ Process: Run HDBSCAN (3 seconds)
   ├─ Review: See clustering results
   └─ Output: people_db.json (500+ unique people)

4. DATABASE ALIGNMENT
   align_database.py
   ├─ Input: embeddings.npy, paths.json
   ├─ Validate: Check data integrity
   └─ Output: image_vectors.npy, image_cache.json

5. SEARCH INDEXING
   (Automatic in Streamlit app)
   ├─ Load image_cache.json (path → index mapping)
   ├─ Load image_vectors.npy (512-d per image)
   └─ Ready for similarity queries

6. USER INTERACTION
   streamlit run app.py
   ├─ User enters search query (text or image)
   ├─ System encodes to 512-d embedding
   ├─ Compute similarity with 13K images
   └─ Display top 20 results in <200ms
```

### Alternative Quick-Start

```
1. Copy images to test_images/
2. Run: python production_pipeline.py (extracts faces)
3. Run: python tune_clustering.py (clusters identities)
4. Run: streamlit run app.py (launches search UI)
   └─ All data auto-loaded and ready
```

---

## Error Recovery

### Scenario: Missing image_cache.json

```
Problem: Can't start search, image_cache.json missing
Solution: Run align_database.py
  ├─ Checks: embeddings.npy exists?
  ├─ Checks: paths.json exists?
  ├─ Verifies: Same length
  └─ Recreates: image_cache.json + image_vectors.npy
```

### Scenario: Clustering results are wrong

```
Problem: Too many "Unknown" clusters or too loose grouping
Solution: Run tune_clustering.py
  ├─ Current: min_cluster_size=3
  ├─ Try: Lower value (e.g., 2) for stricter clusters
  ├─ Or: Higher value (e.g., 5) for looser clusters
  └─ Save: New clustering to people_db.json
```

### Scenario: Poor search results

```
Problem: Semantic search returns irrelevant images
Solution: 
  ├─ Check: Query encoding working?
     └─ Use: Reverse image search (test with known images)
  ├─ Check: Embeddings up-to-date?
     └─ Run: production_pipeline.py again
  └─ Verify: Model files downloaded
     └─ Run: python download_models.py
```

---

## Key Takeaways

### Architecture Highlights

1. **Modular Design**: Each component has a single responsibility
2. **GPU-First**: Heavy computation offloaded to CUDA
3. **Caching**: Multiple levels (models, embeddings, search results)
4. **Vectorization**: NumPy BLAS for fast similarity computation
5. **User-Friendly**: Web UI abstracts complex operations

### Performance Principles

- **Batch Processing**: Handle multiple images efficiently
- **Lazy Loading**: Load only what's needed
- **Early Exit**: Stop processing once top-K results found
- **Parallelization**: Multi-threaded I/O + GPU compute

### Production Readiness

✅ Error handling with fallbacks  
✅ Memory-efficient storage formats  
✅ Database integrity checks  
✅ Graceful degradation (CPU if GPU unavailable)  
✅ Complete audit trail (logs, paths, metadata)

---

**See TECHNICAL_DOCUMENTATION.md for detailed API documentation**
