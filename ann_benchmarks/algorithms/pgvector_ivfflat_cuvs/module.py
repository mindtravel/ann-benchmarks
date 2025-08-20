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

Enhanced with cuVS optimization for GPU-accelerated vector operations.
"""

import os
import subprocess
import sys
import threading
import time
import numpy as np

import pgvector.psycopg
import psycopg

# 添加 cuvs 相关导入
from cuvs.neighbors import ivf_flat
import cupy

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

# cuVS 距离度量映射 ?真的有用吗
CUVS_METRIC_MAP = {
    "angular": "cosine",
    "euclidean": "euclidean"
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


class PGVectorIVFFlatCuvs(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._lists = method_param['lists']  # Number of lists for IVFFlat
        self._batch_size = method_param.get('batch_size', 5000)  # 批处理大小
        self._cur = None
        self._conn = None
        
        # cuVS 相关属性
        self._cuvs_index = None
        self._cuvs_vectors = None
        self._cuvs_ids = None
        self._vector_dim = None

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



    def _build_cuvs_index(self, dataset: np.ndarray) -> None:
        """构建 cuVS 索引"""
        print("Building cuVS index for GPU acceleration...")
        self._vector_dim = dataset.shape[1]
        
        # 将数据转换为 GPU 数组
        self._cuvs_vectors = cupy.asarray(dataset.astype(np.float32))
        self._cuvs_ids = cupy.arange(len(dataset))
        
        # 构建 cuVS IVF-Flat 索引
        cuvs_metric = CUVS_METRIC_MAP.get(self._metric, "euclidean")
        index_params = ivf_flat.IndexParams(n_lists=self._lists, metric=cuvs_metric)
        self._cuvs_index = ivf_flat.build(index_params, self._cuvs_vectors)
        print(f"cuVS IVF-Flat index built with {self._lists} lists")

    def _cuvs_search(self, query_vector: np.ndarray, k: int) -> List[int]:
        """使用 cuVS 执行单个查询"""
        if self._cuvs_index is None:
            raise RuntimeError("cuVS index not available")
            
        # 将查询向量转换为 GPU 数组
        query_gpu = cupy.asarray(np.array([query_vector], dtype=np.float32))
        
        # 执行搜索
        search_params = ivf_flat.SearchParams(n_probes=self._probes)
        distances, indices = ivf_flat.search(search_params, self._cuvs_index, query_gpu, k)
        
        # 转换回 CPU
        indices_cpu = cupy.asnumpy(indices)
        ids_cpu = cupy.asnumpy(self._cuvs_ids)
        
        # 返回对应的 ID
        result_ids = [ids_cpu[idx] for idx in indices_cpu[0]]
        return result_ids

    def _cuvs_batch_search(self, query_vectors: np.ndarray, k: int) -> List[List[int]]:
        """使用 cuVS 执行批量查询 - 真正的 GPU 批量处理"""
        if self._cuvs_index is None:
            raise RuntimeError("cuVS index not available")
            
        print(f"Performing cuVS IVF-Flat batch search for {len(query_vectors)} queries...")
        
        # 将查询向量转换为 GPU 数组
        queries_gpu = cupy.asarray(query_vectors.astype(np.float32))
        
        # 执行批量搜索
        search_params = ivf_flat.SearchParams(n_probes=self._probes)
        distances, indices = ivf_flat.search(search_params, self._cuvs_index, queries_gpu, k)
        
        # 转换回 CPU
        indices_cpu = cupy.asnumpy(indices)
        ids_cpu = cupy.asnumpy(self._cuvs_ids)
        
        # 返回对应的 ID 列表
        results = []
        for i in range(len(query_vectors)):
            result_ids = [ids_cpu[idx] for idx in indices_cpu[i]]
            results.append(result_ids)
        
        print(f"IVF-Flat batch search completed for {len(query_vectors)} queries")
        return results



    def fit(self, dataset):
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
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % dataset.shape[1])
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

        # 构建 cuVS IVF-Flat 索引
        self._build_cuvs_index(dataset)
        print("cuVS IVF-Flat GPU acceleration initialized")

    def set_query_arguments(self, probes):
        self._probes = probes
        print(f"Set cuVS IVF-Flat search probes to {probes}")

    def query(self, v, n):
        """单次查询，使用 cuVS IVF-Flat GPU 加速"""
        return self._cuvs_search(v, n)

    def batch_query(self, X, n):
        """执行批量查询，使用 cuVS IVF-Flat GPU 加速"""
        self._batch_results = self._cuvs_batch_search(X, n)

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
        # cuVS 会自动清理 GPU 资源
        pass

    def __str__(self):
        return f"PGVectorCuvsIVFFlat(lists={self._lists}, probes={self._probes})"