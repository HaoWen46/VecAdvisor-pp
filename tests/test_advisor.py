"""Tests for VecAdvisor++ advisor logic."""

import pytest

from src.advisor.advisor import VecAdvisor
from src.advisor.rules import (
    Recommendation,
    generate_recommendation,
    recommend_hnsw_params,
    recommend_ivfflat_params,
    select_index_type,
)
from src.profiler.workload_profiler import WorkloadProfile, profile_from_params


def _make_profile(**kwargs) -> WorkloadProfile:
    """Create a WorkloadProfile with sensible defaults, overriding with kwargs."""
    defaults = {
        "n_vectors": 1_000_000,
        "dim": 128,
        "k": 10,
        "has_filters": False,
        "filter_selectivity": 1.0,
        "filter_columns": [],
        "update_rate": 0.0,
        "memory_budget_mb": 2048,
        "latency_budget_ms": 50.0,
        "cache_regime": "warm",
    }
    defaults.update(kwargs)
    return WorkloadProfile(**defaults)


class TestIndexTypeSelection:
    def test_small_dataset_no_index(self):
        profile = _make_profile(n_vectors=5000)
        idx_type, _ = select_index_type(profile)
        assert idx_type == "none"

    def test_high_update_rate_ivfflat(self):
        profile = _make_profile(update_rate=0.3)
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_very_selective_filter_ivfflat(self):
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.005,
            filter_columns=["category_1000"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_moderate_filter_hnsw(self):
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.2,
            filter_columns=["category_10"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"

    def test_selective_filter_large_dataset_ivfflat(self):
        """At 1% selectivity on 100K+ dataset, should pick IVFFlat."""
        profile = _make_profile(
            n_vectors=100_000, has_filters=True,
            filter_selectivity=0.01, filter_columns=["category_100"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_5pct_selectivity_large_dataset_ivfflat(self):
        """At 5% selectivity on 1M dataset, should pick IVFFlat."""
        profile = _make_profile(
            n_vectors=1_000_000, has_filters=True,
            filter_selectivity=0.05, filter_columns=["category_100"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_default_hnsw(self):
        profile = _make_profile()
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"

    def test_latency_critical_warm_cache_hnsw(self):
        profile = _make_profile(latency_budget_ms=5.0, cache_regime="warm")
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"


class TestHNSWParams:
    def test_default_params(self):
        profile = _make_profile()
        build, query, _ = recommend_hnsw_params(profile)
        assert build["m"] == 16
        assert build["ef_construction"] == 128
        assert query["ef_search"] >= 40

    def test_high_dim_increases_m(self):
        profile = _make_profile(dim=512)
        build, _, _ = recommend_hnsw_params(profile)
        assert build["m"] == 32

    def test_tight_memory_decreases_m(self):
        profile = _make_profile(memory_budget_mb=512, n_vectors=1_000_000)
        build, _, _ = recommend_hnsw_params(profile)
        assert build["m"] == 8

    def test_filter_increases_ef_search(self):
        profile_no_filter = _make_profile()
        profile_filtered = _make_profile(
            has_filters=True, filter_selectivity=0.05
        )
        _, query_nf, _ = recommend_hnsw_params(profile_no_filter)
        _, query_f, _ = recommend_hnsw_params(profile_filtered)
        assert query_f["ef_search"] > query_nf["ef_search"]

    def test_very_selective_filter_large_ef_search(self):
        profile = _make_profile(has_filters=True, filter_selectivity=0.005)
        _, query, _ = recommend_hnsw_params(profile)
        assert query["ef_search"] >= 400


class TestIVFFlatParams:
    def test_default_params(self):
        profile = _make_profile()
        build, query, _ = recommend_ivfflat_params(profile)
        assert build["lists"] >= 100
        assert query["probes"] >= 1

    def test_large_dataset_more_lists(self):
        profile_small = _make_profile(n_vectors=100_000)
        profile_large = _make_profile(n_vectors=5_000_000)
        build_s, _, _ = recommend_ivfflat_params(profile_small)
        build_l, _, _ = recommend_ivfflat_params(profile_large)
        assert build_l["lists"] > build_s["lists"]

    def test_filter_increases_probes(self):
        profile_nf = _make_profile()
        profile_f = _make_profile(has_filters=True, filter_selectivity=0.01)
        _, query_nf, _ = recommend_ivfflat_params(profile_nf)
        _, query_f, _ = recommend_ivfflat_params(profile_f)
        assert query_f["probes"] > query_nf["probes"]


class TestFullRecommendation:
    def test_generates_recommendation(self):
        profile = _make_profile()
        rec = generate_recommendation(profile)
        assert isinstance(rec, Recommendation)
        assert rec.index_type in ("hnsw", "ivfflat", "none")

    def test_no_index_for_small_dataset(self):
        profile = _make_profile(n_vectors=1000)
        rec = generate_recommendation(profile)
        assert rec.index_type == "none"
        assert len(rec.build_params) == 0

    def test_filtered_includes_auxiliary(self):
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.1,
            filter_columns=["category_10"]
        )
        rec = generate_recommendation(profile)
        assert len(rec.auxiliary_indexes) > 0
        assert rec.auxiliary_indexes[0]["column"] == "category_10"

    def test_session_settings_present(self):
        profile = _make_profile(n_vectors=1_000_000)
        rec = generate_recommendation(profile)
        assert "work_mem" in rec.session_settings
        assert "maintenance_work_mem" in rec.session_settings


class TestVecAdvisorInterface:
    def test_analyze_from_params(self):
        advisor = VecAdvisor()
        rec = advisor.analyze_from_params(
            n_vectors=500_000, dim=128, k=10,
            has_filters=True, filter_selectivity=0.05,
            filter_columns=["category_100"],
        )
        assert rec.index_type in ("hnsw", "ivfflat")
        assert len(rec.explanation) > 0

    def test_get_sql(self):
        advisor = VecAdvisor()
        rec = advisor.analyze_from_params(
            n_vectors=500_000, dim=128, k=10,
        )
        sql = advisor.get_sql(rec, "test_table")
        assert "CREATE INDEX" in sql or "No vector index" in sql
