# Vision Archive AI - Quick Reference & User Guide

---

## Quick Start (5 Minutes)

### Prerequisites
```bash
# Check Python version
python --version  # Requires 3.8+

# Check NVIDIA GPU
nvidia-smi  # Should show GPU memory if CUDA is available
```

### Installation
```bash
# 1. Clone/navigate to project
cd vision_ai_engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -q insightface onnxruntime-gpu numpy opencv-python hdbscan streamlit Pillow tokenizers requests

# 4. Download models (if not present)
python download_models.py  # ~500 MB download
```

### First Run
```bash
# 1. Get test images
python prepare_lfw.py  # Downloads 13K images from LFW dataset

# 2. Extract face embeddings (takes ~1 minute for 13K images)
python production_pipeline.py

# 3. Cluster similar faces (takes ~3 seconds)
python tune_clustering.py
# When prompted, save to database? → yes

# 4. Launch Streamlit interface
streamlit run app.py
# Opens http://localhost:8501 automatically
```

---

## Module Usage Guide

### Core Modules

#### `ai_engine.py` - Face Detection & Embedding

**Direct Usage**:
```python
from ai_engine import AIEngine
import cv2

# Initialize
ai = AIEngine()

# Process image
img = cv2.imread("photo.jpg")
faces = ai.get_faces(img)

# Access results
if faces:
    embedding = faces[0].embedding  # 512-d vector
    bbox = faces[0].bbox  # [x1, y1, x2, y2]
    print(f"Found {len(faces)} faces")
```

**Typical Output**:
```
Found 3 faces
Face 0: Confidence=0.98, Embedding shape=(512,)
Face 1: Confidence=0.85, Embedding shape=(512,)
Face 2: Confidence=0.72, Embedding shape=(512,)
```

---

#### `search_engine.py` - Semantic Search

**Text-to-Image Search**:
```python
from search_engine import SearchEngine
import numpy as np

searcher = SearchEngine()

# Get text embedding
query_vec = searcher.get_text_embedding("man in suit")
# Returns: 512-d normalized vector

# Compare with image database (external)
scores = np.dot(image_database, query_vec)
top_indices = np.argsort(-scores)[:5]
```

**Image-to-Image Search**:
```python
from PIL import Image

img = Image.open("reference.jpg")
query_vec = searcher.get_image_embedding(img)
# Can pass PIL Image or file path (string)
```

---

#### `vector_storage.py` - Embedding Cache

**Loading Cached Embeddings**:
```python
from vector_storage import VectorStorage

storage = VectorStorage()
if storage.load():
    print(f"Loaded {len(storage.data_map)} images")
    # storage.vectors: (N, 512) numpy array
    # storage.data_map: {path: index} dict
```

**Saving New Embeddings**:
```python
storage = VectorStorage()
storage.save(
    paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    vectors=np.array([/* 512-d vectors */])
)
```

---

#### `cluster_engine.py` - Face Clustering

**Cluster Similar Faces**:
```python
from cluster_engine import ClusterEngine
import numpy as np

# Initialize with parameters
clusterer = ClusterEngine(min_cluster_size=3)

# Get embeddings (13K images)
embeddings = np.load("models/embeddings.npy")

# Cluster
labels = clusterer.fit_predict(embeddings)
# Output: [-1, 0, 0, 1, 0, -1, 1, ...]
#         -1 = outlier, 0+ = cluster ID
```

**Interpreting Results**:
```python
unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = list(labels).count(-1)

print(f"Found {unique_clusters} unique people")
print(f"{noise_count} faces marked as unknown")
```

---

## Data Pipeline Scripts

### Face Extraction

#### High-Performance Pipeline
```bash
# Multi-threaded, 200+ FPS on RTX 3050
python production_pipeline.py
```

**Features**:
- Automatic GPU worker scaling
- Parallel image I/O + GPU inference
- Progress bar with FPS counter
- Incremental saving (doesn't reprocess existing images)

**Output**:
```
🚀 VISION ARCHIVE: AUTO-SCALING PIPELINE
Found 1000 new images.
⚙️  Hardware Strategy: 2 Parallel GPU Workers
🔥 Starting Engines... (This takes ~2s)
🚀 Speed: 157.3 FPS | Progress: 500/1000
...
✅ DONE! Avg Speed: 165.4 FPS
```

#### Simple Pipeline
```bash
# Single-threaded, for testing
python test_pipeline_queue.py
```

---

### Clustering & Tuning

#### Interactive Clustering
```bash
python tune_clustering.py
```

**Workflow**:
```
⚡ VISION ARCHIVE: INSTANT TUNER ⚡
Loading 13,000+ face vectors from disk... Done. (13000 faces)

========================================
Enter new Threshold (0.35 - 0.60) or 'exit': 0.45
✅ Re-clustered in 2.87s
   -> Found 523 unique people.
   -> 342 faces marked as 'Unknown/Noise'.

Save this result to Database? (y/n): y
💾 Saved. Run 'python gallery.py' to view the new clusters.

========================================
Enter new Threshold (0.35 - 0.60) or 'exit': exit
```

**Threshold Interpretation**:
- **0.35** (Strict): Many small clusters, more unknowns
- **0.45** (Balanced): ~500 clusters, ~300 unknowns
- **0.60** (Loose): Fewer clusters, more grouping

---

### Database Operations

#### Repair/Align Database
```bash
python align_database.py
```

**Validates**:
```
🔧 VISION ARCHIVE: Database Repair Tool
   -> Found pipeline output (embeddings.npy)
   -> Found pipeline output (paths.json)
✅ Data Validated: 13000 images.
💾 Saving standardized database...
✅ DONE! You can now run 'streamlit run app.py'
```

#### Rename Person
```bash
python rename_person.py Person_5 "John Doe"
```

**Output**:
```
✅ Success: 'Unknown' (Person_5) is now 'John Doe'
```

---

## User Interface Scripts

### Streamlit Web Application

```bash
streamlit run app.py
```

**Features**:
1. **Semantic Search** (Global search with text/image)
2. **Detective Mode** (Search within person's photos)
3. **Face Gallery** (Browse clusters)
4. **System Stats** (Database info)

**Typical Usage**:
```
1. Select: "Semantic Search" mode
2. Choose: "Text Description"
3. Enter: "person smiling"
4. View: Top 20 matching images with scores
5. Click: Any image to inspect further
```

### Command-Line Search

```bash
python search_app.py
```

**Interaction**:
```
🔎 Enter search query (or 'exit'): person in suit
[Top 5 results with similarity scores]
 [0.8234] test_images/person1.jpg
 [0.7891] test_images/person2.jpg
 [0.7234] test_images/person3.jpg

🔎 Enter search query (or 'exit'): exit
```

### HTML Gallery

```bash
python gallery.py
```

**Output**:
```
✅ Gallery generated: gallery.html
🚀 Server started! Open this link: http://localhost:8000/gallery.html
```

Then opens browser automatically to view interactive gallery.

---

## Diagnostic & Debug Tools

### System Benchmarks

```bash
python benchmark.py
```

**Output**:
```
🕵️ VISION ARCHIVE: System Diagnostic Tool
   ✅ Local models found in models/buffalo_s/. Proceeding offline...

1️⃣  TESTING DISK READ SPEED...
   -> Disk Speed: 234.5 images/sec

2️⃣  TESTING CPU DECODE SPEED...
   -> Decode Speed: 89.3 images/sec

3️⃣  TESTING GPU AI SPEED...
   -> AI Inference Speed: 157.2 FPS
```

**Interpretation**:
- **Disk Speed**: ~100-200 typical for SSD
- **Decode Speed**: ~50-150 typical for CV2
- **AI Speed**: 150+ FPS = Good GPU

### Compare Two Faces

```bash
python check_similarity.py
```

**Manually Configured**:
Edit the script to set:
```python
img_path1 = "test_images/Aaron_Peirsol_0001.jpg"
img_path2 = "test_images/Abdoulaye_Wade_0001.jpg"
```

**Output**:
```
Extracting Face 1...
Extracted Face 2...

--- RESULTS ---
Similarity Score: 0.2347 (Higher = More Similar)
Cosine Distance:  0.7653 (Lower = More Similar)

✅ The AI knows these are DIFFERENT people.
   -> Your threshold (eps) should be near this value.
```

---

## Common Workflows

### Workflow 1: Organize New Photo Collection

```bash
# 1. Copy images to test_images/
cp ~/Downloads/vacation_photos/* test_images/

# 2. Extract faces (if not already done)
python production_pipeline.py

# 3. Cluster & tune
python tune_clustering.py

# 4. View results in gallery
python gallery.py

# 5. Rename clusters as needed
python rename_person.py Person_3 "Mom"
python rename_person.py Person_7 "Dad"

# 6. Use web app for detailed search
streamlit run app.py
```

### Workflow 2: Search Existing Database

```bash
# 1. Ensure embeddings are loaded
python align_database.py  # Quick check

# 2. Launch search interface
streamlit run app.py

# 3. Use Semantic Search or Detective Mode
# 4. Optional: Export results for analysis
```

### Workflow 3: Retune Clustering

```bash
# Current results too loose/strict?

# 1. Load the embedding cache
python tune_clustering.py

# 2. Try different thresholds
# 0.45 → Found 523 people, 342 unknown
# 0.50 → Found 485 people, 298 unknown
# Save preferred result

# 3. View updated gallery
python gallery.py
```

---

## Configuration & Tuning

### Performance Tuning

**GPU Memory Limited?**
```python
# In production_pipeline.py, reduce:
DET_SIZE = (240, 240)  # Instead of (320, 320)
NUM_AI_INSTANCES = 1   # Instead of 2
```

**Want Faster Results?**
```python
# In app.py, reduce displayed results:
top_indices = np.argsort(-scores)[:10]  # Instead of [:20]
```

**Clustering Too Loose?**
```bash
python tune_clustering.py
# Try: 0.35 (stricter)
```

**Clustering Too Strict?**
```bash
python tune_clustering.py
# Try: 0.60 (looser)
```

---

## File Reference

### Core Processing Files

| File | Purpose | Command | Time |
|------|---------|---------|------|
| `production_pipeline.py` | Extract embeddings (fast) | `python production_pipeline.py` | ~1min for 13K |
| `test_pipeline_queue.py` | Extract embeddings (simple) | `python test_pipeline_queue.py` | ~2min for 13K |
| `tune_clustering.py` | Cluster faces | `python tune_clustering.py` | ~5s per run |
| `align_database.py` | Database validation | `python align_database.py` | ~1s |

### User Interface Files

| File | Purpose | Command | Output |
|------|---------|---------|--------|
| `app.py` | Main Streamlit web app | `streamlit run app.py` | Web UI (port 8501) |
| `search_app.py` | CLI search tool | `python search_app.py` | Command-line REPL |
| `gallery.py` | HTML gallery | `python gallery.py` | gallery.html (port 8000) |

### Utility Files

| File | Purpose | Command |
|------|---------|---------|
| `download_models.py` | Get CLIP models | `python download_models.py` |
| `prepare_lfw.py` | Copy dataset images | `python prepare_lfw.py` |
| `rename_person.py` | Update person name | `python rename_person.py Person_X "Name"` |
| `benchmark.py` | Performance test | `python benchmark.py` |
| `check_similarity.py` | Compare 2 faces | `python check_similarity.py` |

---

## Troubleshooting Quick Reference

| Problem | Probable Cause | Fix |
|---------|---|---|
| `ModuleNotFoundError: No module named 'insightface'` | Missing dependency | `pip install insightface` |
| `CUDA out of memory` | GPU overloaded | Lower batch size or resolution |
| `No module named 'onnx'` | ONNX Runtime not installed | `pip install onnxruntime-gpu` |
| `models/embeddings.npy not found` | Pipeline not run | `python production_pipeline.py` |
| `people_db.json not found` | Clustering not done | `python tune_clustering.py` |
| `Database mismatch` | Corrupted cache | `python align_database.py` |
| `Low FPS (< 50)` | GPU bottleneck | Reduce `DET_SIZE` to (240, 240) |
| `CLIP models missing` | Download incomplete | `python download_models.py` |

---

## Performance Expectations

### On RTX 3050 (6GB VRAM)

```
Task                    Time            FPS/Throughput
────────────────────────────────────────────
Face Extraction         5-7ms/image     150-200 FPS
CLIP Encoding           50-100ms        10-20 queries/sec
Similarity Match        ~50ms           13K images
Clustering              ~3s             For 13K embeddings
Full Pipeline (13K)     ~1 minute       Total
```

### On CPU Only (No GPU)

```
Face Extraction         ~100ms/image    10 FPS
CLIP Encoding           ~500ms          2 queries/sec
Similarity Match        ~200ms          13K images
Clustering              ~60s            For 13K embeddings
```

**Recommendation**: GPU is strongly recommended for production use.

---

## Key Parameters

### Production Pipeline

```python
NUM_AI_INSTANCES = 2        # Parallel GPU workers
DET_SIZE = (320, 320)       # Detection resolution
BATCH_SIZE = 32             # Image processing batch
```

### Clustering

```python
min_cluster_size = 3        # Minimum group size
min_samples = None          # Defaults to min_cluster_size
metric = 'euclidean'        # Distance metric
```

### Search App

```python
SIMILARITY_THRESHOLD = 0.15 # Minimum match score
TOP_K = 20                  # Show top N results
BATCH_SIZE = 5              # Streamlit grid columns
```

---

## Data Format Specifications

### embeddings.npy
- **Type**: NumPy array (float32)
- **Shape**: (N, 512) where N = number of images
- **Size**: ~2 MB per 1,000 images
- **Content**: Face recognition embeddings

### image_vectors.npy
- **Type**: NumPy array (float32)
- **Shape**: (N, 512)
- **Size**: ~2 MB per 1,000 images  
- **Content**: CLIP image embeddings

### image_cache.json
- **Type**: JSON dictionary
- **Format**: `{"image_path": index_in_npy_array}`
- **Size**: ~1 KB per 1,000 images

### people_db.json
- **Type**: JSON nested structure
- **Format**: `{"Person_ID": {"name": str, "photos": [paths]}}`
- **Editable**: Yes (manually modify names)

---

## Best Practices

### Data Management

✅ **DO:**
- Keep `test_images/` for working data
- Backup `models/people_db.json` (manually edited)
- Use `align_database.py` after any manual changes
- Version control the `people_db.json` file

❌ **DON'T:**
- Modify `.npy` files directly
- Delete images while pipeline is running
- Mix old and new embeddings without re-clustering
- Assume face order is preserved (it's not)

### Performance

✅ **DO:**
- Use `production_pipeline.py` for large datasets
- Run clustering once, then just search
- Cache embeddings between sessions
- Use GPU when available

❌ **DON'T:**
- Re-extract embeddings for every search
- Process images individually
- Store embeddings in JSON (use NumPy)
- Run 10+ workers on 6GB GPU

### Development

✅ **DO:**
- Create backup of `people_db.json`
- Test changes on small datasets first
- Verify output shapes after modifications
- Add error handling for file I/O

❌ **DON'T:**
- Hardcode file paths
- Ignore warnings during initialization
- Skip validation checks
- Mix string and Path objects

---

**For detailed technical information, see TECHNICAL_DOCUMENTATION.md**  
**For architecture details, see ARCHITECTURE_WORKFLOW.md**
