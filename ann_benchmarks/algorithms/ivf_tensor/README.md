# IVF-Tensor: GPU-Accelerated IVF-Flat Search

This module provides a Python wrapper for the CUDA-based IVF-Flat search implementation from pgvector.

## Features

- GPU-accelerated K-means clustering
- GPU-accelerated IVF-Flat search
- Support for both angular (cosine) and euclidean (L2) distance metrics
- CPU fallback implementation using sklearn (for testing without CUDA)

## Current Status

✅ **Basic Python wrapper implemented** - Works with CPU fallback
⏳ **CUDA integration** - Needs C/CUDA wrapper library

## Usage

The module can be used in ann-benchmarks like any other algorithm:

```python
from ann_benchmarks.algorithms.ivf_tensor.module import IVFTensor
import numpy as np

# Initialize
algo = IVFTensor('angular', {
    'n_lists': 100,        # Number of clusters
    'kmeans_iters': 20,    # K-means iterations
    'use_minibatch': False # Use minibatch K-means
})

# Fit the index
X = np.random.rand(10000, 96).astype(np.float32)
algo.fit(X)

# Set query parameters
algo.set_query_arguments(n_probes=10)  # Number of clusters to probe

# Query
result = algo.query(X[0], k=10)
print(result)  # List of indices
```

## CUDA Integration (TODO)

To enable GPU acceleration, you need to:

1. **Compile the CUDA library** from pgvector:
   ```bash
   cd /home/diy/lzx/pgvector
   # Build the shared library that exports ivf_search_pipeline
   ```

2. **Create a C wrapper** that exposes the CUDA functions to Python via ctypes or pybind11

3. **Update module.py** to call the CUDA functions instead of CPU fallback

## Configuration

The algorithm supports the following parameters:

- `n_lists`: Number of clusters (default: sqrt(n) for large datasets, n/1000 for small)
- `kmeans_iters`: Number of K-means iterations (default: 20)
- `use_minibatch`: Use minibatch K-means algorithm (default: False)
- `n_probes`: Number of clusters to probe during search (set via `set_query_arguments`)

## Testing

Test the module:

```bash
cd /home/diy/lzx/ann-benchmarks
python3 -c "from ann_benchmarks.algorithms.ivf_tensor.module import IVFTensor; import numpy as np; algo = IVFTensor('angular', {}); X = np.random.rand(1000, 96).astype(np.float32); algo.fit(X); algo.set_query_arguments(5); print(algo.query(X[0], 10))"
```

## Integration with ann-benchmarks

The algorithm is already registered in `config.yml`. To run benchmarks:

```bash
python run.py --dataset glove-100-angular --algorithm ivf_tensor
```




