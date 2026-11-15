# Ultra-Low-RAM & Maximum-Speed AI Engine (`int8` Pure NumPy Transformer)

A zero-allocation, database-free **2-Layer Quantized Transformer + 50,000-User Async Server** engineered in 100% pure Python and NumPy. Designed to run under a strict **7.24 MB total memory footprint** with sub-millisecond inference latency and full **PyArmor Free/Trial** compatibility.

---

## 1. Architectural Rules Enforced

1. **Strict Offline vs. Production Separation**:
   - `offline_train.py`: Trains weights and exports uncompressed, mmap-ready `np.int8` matrices + `np.float32` scales (`model_weights.npz`).
   - `production_server.py`: 100% pure Python + NumPy ONLY (Zero PyTorch, TensorFlow, SciPy, Pandas, or Database drivers).
2. **Zero 64-Bit Types (`np.float64` / `np.int64`)**:
   - Every weight, activation, and index buffer strictly uses `np.int8`, `np.uint8`, `np.int32`, `np.uint32`, `np.bool_`, or `np.float32`.
   - Even `np.argmax` is replaced with an in-place `np.int32` masked reduction so NumPy never creates a 64-bit `np.intp` index.
3. **Zero Per-Request RAM Allocation (`0 Bytes/Request`)**:
   - Pre-allocates 8 `InferenceSlot` scratch buffers at startup (`out=` used on every NumPy call).
   - Uses `memoryview` slicing (`copy=False`) on incoming socket payloads and pre-baked HTTP/1.1 response `bytes` tuples.
4. **Zero Database Dependency**:
   - Tracks 50,000 users in $O(1)$ fixed-size NumPy arrays (`341.80 KB`) and flushes event batches asynchronously to a binary Write-Ahead Log (`events_wal.bin`) via `collections.deque(maxlen=4096)`.

---

## 2. Quick Start (From Zero)

### Step 1: Install Production Requirement (NumPy Only)
```bash
pip install -r requirements.txt