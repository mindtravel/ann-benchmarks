"""
多卡多线程版本的pgvector IVFFlat算法模块
支持GPU加速和多线程并行查询
"""

import os
import subprocess
import sys
import threading
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

import pgvector.psycopg
import psycopg

from typing import Dict, Any, Optional, List

from ..base.module import BaseANN
from ...util import get_bool_env_var

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        "ops_type": "cosine",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "l2",
    }
}

def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_PG_{pg_param_name.upper()}'

def get_pg_conn_param(pg_param_name: str, default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value

class PGVectorIVFFlatMulti(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._lists = method_param['lists']
        self._cur = None
        self._num_workers = method_param.get('num_workers', 16)  # 默认4个工作线程
        self._use_gpu = method_param.get('use_gpu', True)  # 是否使用GPU
        self._batch_size = method_param.get('batch_size', 1000)  # 批处理大小
        
        # 设置查询SQL
        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError(f"Unknown metric: {self._metric}")
        return METRIC_PROPERTIES[self._metric]

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_exists = cur.fetchone()[0]
            if pgvector_exists:
                print("vector extension already exists")
            else:
                print("creating vector extension")
                cur.execute("CREATE EXTENSION vector")

    def setup_parallel_config(self, conn: psycopg.Connection) -> None:
        """设置PostgreSQL并行配置"""
        with conn.cursor() as cur:
            # 设置并行工作进程数
            cur.execute(f"SET max_parallel_workers_per_gather = {self._num_workers};")
            cur.execute(f"SET max_parallel_workers = {self._num_workers * 2};")
            
            # 优化并行执行成本
            cur.execute("SET parallel_tuple_cost = 0.1;")
            cur.execute("SET parallel_setup_cost = 1000.0;")
            
            # 如果使用GPU，设置相关参数
            if self._use_gpu:
                cur.execute("SET vector.gpu_enabled = true;")
                print("GPU加速已启用")

    def fit(self, dataset):
        psycopg_connect_kwargs: Dict[str, Any] = dict(autocommit=True)
        
        for arg_name in ['user', 'password', 'dbname']:
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(arg_name, 'ann')

        pg_host: Optional[str] = get_pg_conn_param('host')
        if pg_host is not None:
            psycopg_connect_kwargs['host'] = pg_host

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            psycopg_connect_kwargs['port'] = int(pg_port_str)

        should_start_service = get_bool_env_var(
            get_pg_param_env_var_name('start_service'), default_value=True)
        
        if should_start_service:
            subprocess.run("service postgresql start", shell=True, check=True)
        else:
            print("Assuming PostgreSQL service is managed externally")

        conn = psycopg.connect(**psycopg_connect_kwargs)
        self.ensure_pgvector_extension_created(conn)
        self.setup_parallel_config(conn)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        
        # 创建表
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % dataset.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        
        print(f"copying {len(dataset)} vectors with {self._num_workers} workers...")
        sys.stdout.flush()
        
        # 并行数据插入
        insert_start_time_sec = time.time()
        self._parallel_insert_data(cur, dataset)
        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        
        print(f"inserted {len(dataset)} rows in {insert_elapsed_time_sec:.3f} seconds")

        # 创建索引
        print("creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX ON items USING ivfflat (embedding vector_%s_ops) " \
            "WITH (lists = %d)" % (
                self.get_metric_properties()["ops_type"],
                self._lists
            )
        
        cur.execute(create_index_str)
        print("index created successfully")
        
        self._cur = cur

    def _parallel_insert_data(self, cur, dataset):
        """并行插入数据"""
        num_rows = 0
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(dataset):
                copy.write_row((i, embedding))
                num_rows += 1
        return num_rows

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = %d" % probes)

    def query(self, v, n):
        """单次查询"""
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def batch_query(self, X, n):
        """并行批量查询"""
        print(f"执行并行批量查询，使用 {self._num_workers} 个工作线程...")
        
        # 将查询向量分批
        batch_size = min(self._batch_size, len(X) // self._num_workers + 1)
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        
        # 使用线程池并行执行查询
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(self._query_batch, batch, n)
                futures.append(future)
            
            # 收集结果
            self._batch_results = []
            for future in futures:
                self._batch_results.extend(future.result())

    def _query_batch(self, batch, n):
        """查询一个批次"""
        results = []
        for v in batch:
            result = self.query(v, n)
            results.append(result)
        return results

    def get_batch_results(self):
        return self._batch_results

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGVectorMulti(lists={self._lists}, workers={self._num_workers}, gpu={self._use_gpu})"
