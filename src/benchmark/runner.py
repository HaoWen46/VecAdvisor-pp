"""Benchmark execution engine for VecAdvisor++.

Runs query workloads against PostgreSQL, measures latency,
and collects system metrics.
"""

import subprocess
import time

import psycopg2
from pgvector.psycopg2 import register_vector

from src.benchmark.metrics import (
    BenchmarkResult,
    compute_latency_percentiles,
    compute_recall,
    compute_topk_completion_rate,
)
from src.benchmark.workload import Query
from src.data.schema import IndexConfig, create_index, get_table_size


class BenchmarkRunner:
    """Executes benchmark workloads and collects performance metrics."""

    def __init__(self, conn_params: dict):
        """Initialize with database connection parameters.

        Args:
            conn_params: Dict with host, port, dbname, user, password.
        """
        self.conn_params = conn_params

    def _get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        register_vector(conn)
        return conn

    def set_query_params(self, conn, index_type: str, params: dict) -> None:
        """Set session-level query parameters for vector search.

        Args:
            conn: psycopg2 connection.
            index_type: "hnsw" or "ivfflat".
            params: Query-time parameters (e.g., ef_search, probes).
        """
        with conn.cursor() as cur:
            if index_type == "hnsw":
                ef_search = params.get("ef_search", 40)
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            elif index_type == "ivfflat":
                probes = params.get("probes", 10)
                cur.execute(f"SET ivfflat.probes = {probes};")

    def run_build_benchmark(
        self, table_name: str, config: IndexConfig
    ) -> tuple[str, float]:
        """Build an index and measure build time.

        Args:
            table_name: Table to index.
            config: Index configuration.

        Returns:
            Tuple of (index_name, build_time_seconds).
        """
        conn = self._get_connection()
        try:
            index_name, build_time = create_index(conn, table_name, config)
            return index_name, build_time
        finally:
            conn.close()

    def run_query_benchmark(
        self,
        queries: list[Query],
        index_type: str,
        query_params: dict,
        warmup_queries: int = 100,
        cache_mode: str = "warm",
    ) -> tuple[list[list[int]], list[float]]:
        """Execute queries and measure per-query latency.

        Args:
            queries: List of Query objects.
            index_type: "hnsw" or "ivfflat".
            query_params: Session parameters (ef_search, probes).
            warmup_queries: Number of warmup queries before measurement.
            cache_mode: "cold" or "warm".

        Returns:
            Tuple of (result_ids, latencies_ms):
                - result_ids: List of lists of returned row IDs per query.
                - latencies_ms: Per-query latency in ms.
        """
        if cache_mode == "cold":
            self._clear_caches()

        conn = self._get_connection()
        try:
            self.set_query_params(conn, index_type, query_params)

            # Warmup
            if cache_mode == "warm":
                warmup_n = min(warmup_queries, len(queries))
                with conn.cursor() as cur:
                    for q in queries[:warmup_n]:
                        cur.execute(q.sql, q.params)
                        cur.fetchall()

            # Measured queries
            result_ids = []
            latencies_ms = []

            with conn.cursor() as cur:
                for q in queries:
                    start = time.perf_counter()
                    cur.execute(q.sql, q.params)
                    rows = cur.fetchall()
                    elapsed = (time.perf_counter() - start) * 1000

                    ids = [row[0] for row in rows]
                    result_ids.append(ids)
                    latencies_ms.append(elapsed)

            return result_ids, latencies_ms
        finally:
            conn.close()

    def run_full_benchmark(
        self,
        table_name: str,
        queries: list[Query],
        index_config: IndexConfig,
        query_params: dict,
        ground_truth_ids,
        k: int,
        config_name: str = "",
        cache_mode: str = "warm",
        filter_selectivity: float | None = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark: build index, run queries, compute metrics.

        Args:
            table_name: Table to benchmark.
            queries: Query workload.
            index_config: Index build configuration.
            query_params: Query-time parameters.
            ground_truth_ids: Ground truth neighbor indices for recall.
            k: Top-k value.
            config_name: Label for this configuration.
            cache_mode: "cold" or "warm".
            filter_selectivity: Optional selectivity for reporting.

        Returns:
            BenchmarkResult with all metrics.
        """
        # Build index
        _, build_time = self.run_build_benchmark(table_name, index_config)

        # Get sizes
        conn = self._get_connection()
        try:
            sizes = get_table_size(conn, table_name)
        finally:
            conn.close()

        # Run queries
        result_ids, latencies_ms = self.run_query_benchmark(
            queries, index_config.index_type, query_params,
            cache_mode=cache_mode,
        )

        # Compute metrics
        recall = compute_recall(result_ids, ground_truth_ids, k)
        lat = compute_latency_percentiles(latencies_ms)
        completion = compute_topk_completion_rate(result_ids, k)

        return BenchmarkResult(
            config_name=config_name,
            index_type=index_config.index_type,
            index_params=index_config.params,
            query_params=query_params,
            recall=recall,
            latency_p50_ms=lat["p50"],
            latency_p95_ms=lat["p95"],
            latency_p99_ms=lat["p99"],
            latency_mean_ms=lat["mean"],
            build_time_s=build_time,
            memory_mb=sizes["index_size_mb"],
            disk_mb=sizes["total_size_mb"],
            completion_rate=completion,
            num_queries=len(queries),
            k=k,
            filter_selectivity=filter_selectivity,
        )

    def _clear_caches(self) -> None:
        """Attempt to clear OS page cache for cold-cache benchmarks.

        On macOS, uses 'purge'. Falls back silently if not available.
        """
        try:
            subprocess.run(["sudo", "purge"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not clear OS caches. Cold-cache results may be inaccurate.")

        # Also restart PostgreSQL shared buffers by reconnecting
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("DISCARD ALL;")
            conn.close()
        except Exception:
            pass
