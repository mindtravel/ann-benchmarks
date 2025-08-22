"""
This module supports connecting to a PostgreSQL instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of PostgreSQL user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If PostgreSQL is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting PostgreSQL
connection parameters:

ANN_BENCHMARKS_PG_USER
ANN_BENCHMARKS_PG_PASSWORD
ANN_BENCHMARKS_PG_DBNAME
ANN_BENCHMARKS_PG_HOST
ANN_BENCHMARKS_PG_PORT

This module starts the PostgreSQL service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_PG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.

This is a multi-threaded version that uses thread pools for parallel query processing.
"""

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
import random

import pgvector.psycopg
import psycopg

from typing import Dict, Any, Optional, List, Tuple

from ..base.module import BaseANN
from ...util import get_bool_env_var

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<=>",
        # A substring of e.g. vector_cosine_ops or halfvec_cosine_ops
        "ops_type": "cosine",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "l2",
    }
}


def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_PG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value

class IndexingProgressMonitor:
    """
    Continuously logs indexing progress, elapsed and estimated remaining
    indexing time.
    """

    MONITORING_DELAY_SEC = 0.5

    def __init__(self, psycopg_connect_kwargs: Dict[str, str]) -> None:
        self.psycopg_connect_kwargs = psycopg_connect_kwargs
        self.monitoring_condition = threading.Condition()
        self.stop_requested = False
        self.psycopg_connect_kwargs = psycopg_connect_kwargs
        self.prev_phase = None
        self.prev_progress_pct = None
        self.prev_tuples_done = None
        self.prev_report_time_sec = None
        self.time_to_load_all_tuples_sec = None

    def report_progress(
            self,
            phase: str,
            progress_pct: Any,
            tuples_done: Any) -> None:
        if progress_pct is None:
            progress_pct = 0.0
        progress_pct = float(progress_pct)
        if tuples_done is None:
            tuples_done = 0
        tuples_done = int(tuples_done)
        if (phase == self.prev_phase and
                progress_pct == self.prev_progress_pct):
            return
        time_now_sec = time.time()

        elapsed_time_sec = time_now_sec - self.indexing_start_time_sec
        fields = [
            f"Phase: {phase}",
            f"progress: {progress_pct:.1f}%",
            f"elapsed time: {elapsed_time_sec:.3f} sec"
        ]
        if (self.prev_report_time_sec is not None and
            self.prev_tuples_done is not None and
            elapsed_time_sec):
            overall_tuples_per_sec = tuples_done / elapsed_time_sec
            fields.append(
                f"overall tuples/sec: {overall_tuples_per_sec:.2f}")

            time_since_last_report_sec = time_now_sec - self.prev_report_time_sec
            if time_since_last_report_sec > 0:
                cur_tuples_per_sec = ((tuples_done - self.prev_tuples_done) /
                                      time_since_last_report_sec)
                fields.append(
                    f"current tuples/sec: {cur_tuples_per_sec:.2f}")

        remaining_pct = 100 - progress_pct
        if progress_pct > 0 and remaining_pct > 0:
            estimated_remaining_time_sec = \
                elapsed_time_sec / progress_pct * remaining_pct
            estimated_total_time_sec = \
                elapsed_time_sec + estimated_remaining_time_sec
            fields.extend([
                "estimated remaining time: " \
                   f"{estimated_remaining_time_sec:.3f} sec" ,
                f"estimated total time: {estimated_total_time_sec:.3f} sec"
            ])
        print(", ".join(fields))
        sys.stdout.flush()

        self.prev_progress_pct = progress_pct
        self.prev_phase = phase
        self.prev_tuples_done = tuples_done
        self.prev_report_time_sec = time_now_sec

    def monitoring_loop_impl(self, monitoring_cur) -> None:
        while True:
            # Indexing progress query taken from
            # https://github.com/pgvector/pgvector/blob/master/README.md
            monitoring_cur.execute(
                "SELECT phase, " +
                "round(100.0 * blocks_done / nullif(blocks_total, 0), 1), " +
                "tuples_done " +
                "FROM pg_stat_progress_create_index");
            result_rows = monitoring_cur.fetchall()

            if len(result_rows) == 1:
                phase, progress_pct, tuples_done = result_rows[0]
                self.report_progress(phase, progress_pct, tuples_done)
                if (self.time_to_load_all_tuples_sec is None and
                    phase == 'building index: loading tuples' and
                    progress_pct is not None and
                    float(progress_pct) > 100.0 - 1e-7):
                    # Even after pgvector reports progress as 100%, it still spends
                    # some time postprocessing the index and writing it to disk.
                    # We keep track of the the time it takes to reach 100%
                    # separately.
                    self.time_to_load_all_tuples_sec = \
                        time.time() - self.indexing_start_time_sec
            elif len(result_rows) > 0:
                # This should not happen.
                print(f"Expected exactly one progress result row, got: {result_rows}")
            with self.monitoring_condition:
                if self.stop_requested:
                    return
                self.monitoring_condition.wait(
                    timeout=self.MONITORING_DELAY_SEC)
                if self.stop_requested:
                    return

    def monitor_progress(self) -> None:
        prev_phase = None
        prev_progress_pct = None
        with psycopg.connect(**self.psycopg_connect_kwargs) as monitoring_conn:
            with monitoring_conn.cursor() as monitoring_cur:
                self.monitoring_loop_impl(monitoring_cur)

    def start_monitoring_thread(self) -> None:
        self.indexing_start_time_sec = time.time()
        self.monitoring_thread = threading.Thread(target=self.monitor_progress)
        self.monitoring_thread.start()

    def stop_monitoring_thread(self) -> None:
        with self.monitoring_condition:
            self.stop_requested = True
            self.monitoring_condition.notify_all()
        self.monitoring_thread.join()
        self.indexing_time_sec = time.time() - self.indexing_start_time_sec

    def report_timings(self) -> None:
        print(f"pgvector total indexing time: {self.indexing_time_sec:3f} sec")
        if self.time_to_load_all_tuples_sec is not None:
            print("    Time to load all tuples into the index: {:.3f} sec".format(
                self.time_to_load_all_tuples_sec
            ))
            postprocessing_time_sec = \
                self.indexing_time_sec - self.time_to_load_all_tuples_sec
            print("    Index postprocessing time: {:.3f} sec".format(
                postprocessing_time_sec))
        else:
            print("    Detailed breakdown of indexing time not available.")


class PGVectorIVFMulti(BaseANN):
    def __init__(self, metric, method_param):
        self._lists = method_param['lists']  # Number of lists for IVF
        self._num_workers = method_param.get('num_workers', 4)  # 线程池大小
        self._batch_size = method_param.get('batch_size', 5000)  # 批处理大小
        self._cur = None
        self._index_type = method_param.get('index_type', 'flat')  # choice=["flat", "jl", pq"]
        
        if self._index_type == 'jl':
            self._metric = 'euclidean'
        else:
            self._metric = metric
            
        self._target_dim = method_param.get('target_dim', 64)  # JL降维目标维度
        self._enable_reordering = method_param.get('enable_reordering', False)  # 是否启用重排序
        self._candidate_multiplier = method_param.get('candidate_multiplier', 10)  # 候选向量倍数
        
        self._conn = None
        self._thread_pool = None
        self._connection_pool = Queue()
        self._lock = threading.Lock()
        
        # JL相关属性
        self._projection_matrix = None
        self._original_dataset = None
        self._original_dim = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        """
        Get properties of the metric type associated with this index.

        Returns:
            A dictionary with keys distance_operator and ops_type.
        """
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError(
                "Unknown metric: {}. Valid metrics: {}".format(
                    self._metric,
                    ', '.join(sorted(METRIC_PROPERTIES.keys()))
                ))
        return METRIC_PROPERTIES[self._metric]

    def ensure_pgvector_extension_created(self, conn: psycopg.Connection) -> None:
        """
        Ensure that `CREATE EXTENSION vector` has been executed.
        """
        with conn.cursor() as cur:
            # We have to use a separate cursor for this operation.
            # If we reuse the same cursor for later operations, we might get
            # the following error:
            # KeyError: "couldn't find the type 'vector' in the types registry"
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_exists = cur.fetchone()[0]
            if pgvector_exists:
                print("vector extension already exists")
            else:
                print("vector extension does not exist, creating")
                cur.execute("CREATE EXTENSION vector")

    def _create_connection(self) -> psycopg.Connection:
        """创建新的数据库连接"""
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')

        pg_host: Optional[str] = get_pg_conn_param('host')
        if pg_host is not None:
            psycopg_connect_kwargs['host'] = pg_host

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            psycopg_connect_kwargs['port'] = int(pg_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)
        pgvector.psycopg.register_vector(conn)
        return conn

    def _get_connection(self) -> psycopg.Connection:
        """从连接池获取连接"""
        try:
            return self._connection_pool.get_nowait()
        except:
            return self._create_connection()

    def _return_connection(self, conn: psycopg.Connection) -> None:
        """将连接返回到连接池"""
        try:
            self._connection_pool.put_nowait(conn)
        except:
            conn.close()

    def _generate_jl_projection_matrix(self, original_dim: int, target_dim: int) -> np.ndarray:
        """
        生成JL随机投影矩阵
        使用稀疏三值分布：{-1, 0, +1}，概率分别为{1/6, 2/3, 1/6}
        """
        print(f"Generating JL projection matrix: {original_dim} -> {target_dim}")
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 生成随机数
        r = np.random.random((original_dim, target_dim))
        
        # 应用三值分布
        projection = np.zeros((original_dim, target_dim))
        projection[r < 1/6] = -1.0
        projection[r > 5/6] = 1.0
        
        return projection

    def _apply_jl_reduction(self, data: np.ndarray) -> np.ndarray:
        """
        应用JL降维
        """
        if self._projection_matrix is None:
            raise RuntimeError("Projection matrix not initialized")
        
        # 执行矩阵乘法进行降维
        reduced_data = data @ self._projection_matrix
        return reduced_data

    def _normalize_vectors(self, data: np.ndarray) -> np.ndarray:
        """
        对向量进行L2标准化
        """
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        print(f"normalize: {data.shape}, {norms.shape}")

        # 避免除零
        norms = np.where(norms < 1e-8, 1.0, norms)
        # print(data[0], norms[0], data[0] / norms[0])
        return data / norms

    def _query_worker(self, args: Tuple[np.ndarray, int, int]) -> Tuple[int, List[int]]:
        """工作线程函数，执行单个查询"""
        vector, n, query_id = args
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(self._query, (vector, n), binary=True, prepare=True)
                result = [id for id, in cur.fetchall()]
                return query_id, result
        finally:
            self._return_connection(conn)

    def _query_worker_with_reordering(self, args: Tuple[np.ndarray, np.ndarray, int, int]) -> Tuple[int, List[int]]:
        """带重排序的工作线程函数"""
        original_vector, reduced_vector, n, query_id = args
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # 获取更多候选向量
                candidate_n = n * self._candidate_multiplier
                cur.execute(self._query, (reduced_vector, candidate_n), binary=True, prepare=True)
                candidates = [id for id, in cur.fetchall()]
                print(candidates)
                
                # 重排序：使用原始高维向量计算距离
                reordered_candidates = []
                for candidate_id in candidates:# self._metric = "euclidean"
                    # 计算原始高维空间中的距离
                    distance = np.linalg.norm(original_vector - self._original_dataset[candidate_id])
                    reordered_candidates.append((candidate_id, distance))
                
                # 按距离排序并取前n个
                print(f"before:{reordered_candidates}")
                reordered_candidates.sort(key=lambda x: x[1])
                print(reordered_candidates)
                
                result = [id for id, _ in reordered_candidates[:n]]
                return query_id, result
        finally:
            self._return_connection(conn)

    def fit(self, dataset):
        self._original_dataset = dataset.copy()
        self._original_dim = dataset.shape[1]
        
        # 根据索引类型处理数据
        if self._index_type == 'jl':
            print(f"Applying JL dimensionality reduction: {self._original_dim} -> {self._target_dim}")
            
            # 生成投影矩阵
            self._projection_matrix = self._generate_jl_projection_matrix(self._original_dim, self._target_dim)
            
            # 降维+标准化
            dataset = self._apply_jl_reduction(dataset)
            dataset = self._normalize_vectors(dataset)
            
        else:
            self._target_dim = self._original_dim

        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        pg_host: Optional[str] = get_pg_conn_param('host')
        if pg_host is not None:
            psycopg_connect_kwargs['host'] = pg_host

        pg_port_str: Optional[str] = get_pg_conn_param('port')
        if pg_port_str is not None:
            psycopg_connect_kwargs['port'] = int(pg_port_str)

        should_start_service = get_bool_env_var(
            get_pg_param_env_var_name('start_service'),
            default_value=True)
        if should_start_service:
            subprocess.run(
                "service postgresql start",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr)
        else:
            print(
                "Assuming that PostgreSQL service is managed externally. "
                "Not attempting to start the service.")

        self._conn = psycopg.connect(**psycopg_connect_kwargs)
        self.ensure_pgvector_extension_created(self._conn)

        pgvector.psycopg.register_vector(self._conn)
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % self._target_dim)
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(dataset):
                copy.write_row((i, embedding))
                num_rows += 1
        insert_elapsed_time_sec = time.time() - insert_start_time_sec
        print("inserted {} rows into table in {:.3f} seconds".format(
            num_rows, insert_elapsed_time_sec))

        print("creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX ON items USING ivfflat (embedding vector_%s_ops) " \
            "WITH (lists = %d)" % (
                self.get_metric_properties()["ops_type"],
                self._lists
            )
        progress_monitor = IndexingProgressMonitor(psycopg_connect_kwargs)
        progress_monitor.start_monitoring_thread()

        try:
            cur.execute(create_index_str)
        finally:
            progress_monitor.stop_monitoring_thread()
        print("done!")
        progress_monitor.report_timings()
        self._cur = cur

        # 初始化线程池和连接池
        print(f"Initializing thread pool with {self._num_workers} workers")
        self._thread_pool = ThreadPoolExecutor(max_workers=self._num_workers)
        
        for _ in range(self._num_workers):
            conn = self._create_connection()
            self._connection_pool.put(conn)

    def set_query_arguments(self, probes):
        self._probes = probes
        # 为所有连接设置probes参数
        connections = []
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                connections.append(conn)
            except:
                break
        
        for conn in connections:
            with conn.cursor() as cur:
                cur.execute("SET ivfflat.probes = %d" % probes)
            self._connection_pool.put(conn)

    def query(self, v, n):
        """单次查询"""
        if self._index_type == 'jl':
            v_reduced = self._apply_jl_reduction(v.reshape(1, -1)).flatten()
            v_reduced = self._normalize_vectors(v_reduced.reshape(1, -1)).flatten()
            
            if self._enable_reordering:
                # 带重排序的查询
                return self._query_with_reordering(v, v_reduced, n)
            else:
                # 直接使用降维向量查询
                self._cur.execute(self._query, (v_reduced, n), binary=True, prepare=True)
                return [id for id, in self._cur.fetchall()]
        else:
            # 原始查询
            self._cur.execute(self._query, (v, n), binary=True, prepare=True)
            return [id for id, in self._cur.fetchall()]

    def _query_with_reordering(self, original_vector, reduced_vector, n):
        """带重排序的查询"""
        # 获取更多候选向量
        candidate_n = n * self._candidate_multiplier
        self._cur.execute(self._query, (reduced_vector, candidate_n), binary=True, prepare=True)
        candidates = [id for id, in self._cur.fetchall()]
        
        # 重排序：使用原始高维向量计算距离
        reordered_candidates = []
        for candidate_id in candidates:# self._metric = "euclidean"
            # 计算原始高维空间中的距离
            distance = np.linalg.norm(original_vector - self._original_dataset[candidate_id])
            reordered_candidates.append((candidate_id, distance))
        
        # 按距离排序并取前n个
        reordered_candidates.sort(key=lambda x: x[1])
        
        return [id for id, _ in reordered_candidates[:n]]

    def batch_query(self, X, n):
        """执行批量查询，使用线程池并行处理"""
        if self._thread_pool is None:
            raise RuntimeError("Thread pool not initialized. Call fit() first.")
        batch_query_begin = time.time()
        
        num_queries = len(X)
        print(f"Executing batch query with {num_queries} queries using {self._num_workers} workers")
        
        if self._index_type == 'jl':
            # 对查询向量进行降维
            X_reduced = self._apply_jl_reduction(X)
            X_reduced = self._normalize_vectors(X_reduced)
            
            if self._enable_reordering:
                # 带重排序的批量查询
                tasks = []
                for i, (original_vector, reduced_vector) in enumerate(zip(X, X_reduced)):
                    tasks.append((original_vector, reduced_vector, n, i))
                
                futures = []
                for task in tasks:
                    future = self._thread_pool.submit(self._query_worker_with_reordering, task)
                    futures.append(future)
            else:
                # 直接使用降维向量的批量查询
                tasks = []
                for i, vector in enumerate(X_reduced):
                    tasks.append((vector, n, i))
                
                futures = []
                for task in tasks:
                    future = self._thread_pool.submit(self._query_worker, task)
                    futures.append(future)
        else:
            # 原始批量查询
            tasks = []
            for i, vector in enumerate(X):
                tasks.append((vector, n, i))
            
            futures = []
            for task in tasks:
                future = self._thread_pool.submit(self._query_worker, task)
                futures.append(future)
        
        # 收集结果
        results = [None] * num_queries
        for future in as_completed(futures):
            try:
                query_id, result = future.result()
                results[query_id] = result
            except Exception as e:
                print(f"Error in query worker: {e}")
                raise
        
        self._time_to_finish_a_query = (time.time() - batch_query_begin)/num_queries
        self._batch_results = results
        print(self._time_to_finish_a_query)

    def get_batch_results(self):
        """获取批量查询的结果"""
        return self._batch_results

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __del__(self):
        """清理资源"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        # 关闭连接池中的所有连接
        while not self._connection_pool.empty():
            try:
                conn = self._connection_pool.get_nowait()
                conn.close()
            except:
                pass

    def __str__(self):
        index_info = f"index_type={self._index_type}"
        if self._index_type == 'jl':
            index_info += f",{self._original_dim}=>{self._target_dim}"
            if self._enable_reordering:
                index_info += f",reordering={self._candidate_multiplier}X"
        
        return f"PGVectorMulti(lists={self._lists}, probes={self._probes}, workers={self._num_workers}, {index_info})"