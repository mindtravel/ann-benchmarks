#!/usr/bin/env python3

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing module imports...")

try:
    print("1. Testing basic imports...")
    import os
    import subprocess
    import sys
    import threading
    import time
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import numpy as np
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")

try:
    print("2. Testing pgvector imports...")
    import pgvector.psycopg
    import psycopg
    print("✓ pgvector imports successful")
except Exception as e:
    print(f"✗ pgvector imports failed: {e}")

try:
    print("3. Testing ann_benchmarks imports...")
    from ann_benchmarks.algorithms.base.module import BaseANN
    from ann_benchmarks.util import get_bool_env_var
    print("✓ ann_benchmarks imports successful")
except Exception as e:
    print(f"✗ ann_benchmarks imports failed: {e}")

try:
    print("4. Testing module import...")
    from ann_benchmarks.algorithms.pgvector_ivfflat_multi_ours import module
    print("✓ Module import successful")
except Exception as e:
    print(f"✗ Module import failed: {e}")

try:
    print("5. Testing class instantiation...")
    from ann_benchmarks.algorithms.pgvector_ivfflat_multi_ours.module import PGVectorIVFFlatMulti
    algo = PGVectorIVFFlatMulti("angular", {"lists": 100, "num_workers": 4, "use_gpu": True})
    print("✓ Class instantiation successful")
except Exception as e:
    print(f"✗ Class instantiation failed: {e}")

print("Test completed.")



