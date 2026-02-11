"""Decision rules for VecAdvisor++ index selection and parameter tuning.

Grounded in findings from:
- ICDE 2024: Vector data management in relational DBMSs
- PVLDB 2025: Turbocharging Vector Databases Using Modern SSDs
- pgvector documentation and known failure modes
"""

import math
from dataclasses import dataclass, field

from src.profiler.workload_profiler import WorkloadProfile


@dataclass
class Recommendation:
    """Complete index recommendation from the advisor."""

    index_type: str                        # "hnsw", "ivfflat", or "none"
    build_params: dict = field(default_factory=dict)
    query_params: dict = field(default_factory=dict)
    auxiliary_indexes: list[dict] = field(default_factory=list)
    session_settings: dict = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)


def select_index_type(profile: WorkloadProfile) -> tuple[str, str]:
    """Select the best index type based on workload characteristics.

    Args:
        profile: WorkloadProfile with workload features.

    Returns:
        Tuple of (index_type, reason).
    """
    n = profile.n_vectors
    sel = profile.filter_selectivity

    # Small datasets: sequential scan is sufficient
    if n < 10_000:
        return "none", (
            f"Dataset size ({n}) is small enough for sequential scan. "
            f"Index overhead not justified."
        )

    # High write rate: IVFFlat has cheaper rebuilds
    if profile.update_rate > 0.2:
        return "ivfflat", (
            f"High update rate ({profile.update_rate:.0%}) favors IVFFlat "
            f"due to cheaper index rebuilds."
        )

    # Selective filters: IVFFlat handles better with increased probes
    # Research (ICDE 2024) shows HNSW graph traversal struggles when most
    # neighbors are filtered out — the graph can't find enough candidates.
    # At selectivity <= 10%, IVFFlat with high probes is more reliable.
    if profile.has_filters and sel <= 0.10 and n >= 50_000:
        return "ivfflat", (
            f"Selective filter ({sel:.1%}) on large dataset ({n:,}) favors IVFFlat. "
            f"HNSW graph traversal struggles when most neighbors are filtered out, "
            f"leading to low recall and incomplete top-k results."
        )

    if profile.has_filters and sel < 0.01:
        return "ivfflat", (
            f"Very selective filter ({sel:.1%}) favors IVFFlat. "
            f"HNSW graph traversal struggles when most neighbors are filtered out."
        )

    # Memory-constrained environments
    # HNSW memory ≈ n * dim * 4 * (1 + m/8) bytes roughly
    hnsw_mem_est_mb = n * profile.dim * 4 * 1.5 / (1024 * 1024)
    if hnsw_mem_est_mb > profile.memory_budget_mb * 0.8:
        return "ivfflat", (
            f"HNSW estimated memory ({hnsw_mem_est_mb:.0f} MB) exceeds "
            f"80% of budget ({profile.memory_budget_mb} MB). IVFFlat is smaller."
        )

    # Moderate filters with large dataset: HNSW handles well only above 10%
    if profile.has_filters and sel > 0.1 and n > 100_000:
        return "hnsw", (
            f"Moderate filter selectivity ({sel:.0%}) with large dataset "
            f"({n:,}) suits HNSW's superior recall-latency tradeoff."
        )

    # Latency-critical, warm cache: HNSW is faster
    if profile.cache_regime == "warm" and profile.latency_budget_ms < 10:
        return "hnsw", (
            f"Tight latency budget ({profile.latency_budget_ms} ms) in warm cache "
            f"favors HNSW's faster query processing."
        )

    # Default: HNSW generally provides better recall-latency tradeoff
    return "hnsw", (
        "HNSW selected as default: generally better recall-latency tradeoff "
        "than IVFFlat for most workloads."
    )


def recommend_hnsw_params(profile: WorkloadProfile) -> tuple[dict, dict, list[str]]:
    """Recommend HNSW build and query parameters.

    Args:
        profile: WorkloadProfile.

    Returns:
        Tuple of (build_params, query_params, explanations).
    """
    explanations = []

    # --- Build parameter: m (max connections per node) ---
    m = 16  # default
    if profile.dim > 256:
        m = 32
        explanations.append(f"m=32: high dimensionality ({profile.dim}) needs more connections.")
    elif profile.memory_budget_mb < 1024 and profile.n_vectors > 500_000:
        m = 8
        explanations.append(f"m=8: memory budget ({profile.memory_budget_mb} MB) is tight.")
    else:
        explanations.append("m=16: standard default for moderate dimensionality.")

    # --- Build parameter: ef_construction ---
    ef_construction = 128  # default
    if profile.latency_budget_ms < 5:
        ef_construction = 256
        explanations.append(
            "ef_construction=256: high recall needed for tight latency budget "
            "(better index quality reduces search effort)."
        )
    elif profile.n_vectors > 5_000_000:
        ef_construction = 200
        explanations.append(
            "ef_construction=200: large dataset benefits from higher build quality."
        )
    else:
        explanations.append("ef_construction=128: standard default.")

    # --- Query parameter: ef_search ---
    ef_search = max(profile.k * 4, 40)
    explanations.append(f"ef_search base: max(k*4, 40) = {ef_search}")

    if profile.has_filters:
        sel = profile.filter_selectivity
        if sel < 0.01:
            # Very selective: need to visit many more candidates
            ef_search = min(max(ef_search * 8, 400), 1000)
            explanations.append(
                f"ef_search increased 8x for very selective filter ({sel:.1%}). "
                f"Set to {ef_search}."
            )
        elif sel < 0.05:
            ef_search = min(ef_search * 4, 800)
            explanations.append(
                f"ef_search increased 4x for selective filter ({sel:.1%}). "
                f"Set to {ef_search}."
            )
        elif sel < 0.1:
            ef_search = min(ef_search * 2, 400)
            explanations.append(
                f"ef_search doubled for moderate filter ({sel:.0%}). "
                f"Set to {ef_search}."
            )

    build_params = {"m": m, "ef_construction": ef_construction}
    query_params = {"ef_search": ef_search}

    return build_params, query_params, explanations


def recommend_ivfflat_params(profile: WorkloadProfile) -> tuple[dict, dict, list[str]]:
    """Recommend IVFFlat build and query parameters.

    Args:
        profile: WorkloadProfile.

    Returns:
        Tuple of (build_params, query_params, explanations).
    """
    explanations = []
    n = profile.n_vectors

    # --- Build parameter: lists (number of Voronoi cells) ---
    lists = max(100, int(math.sqrt(n)))
    if n > 1_000_000:
        lists = min(int(4 * math.sqrt(n)), 10000)
        explanations.append(
            f"lists={lists}: large dataset ({n:,}), using 4*sqrt(n) capped at 10000."
        )
    else:
        explanations.append(f"lists={lists}: sqrt(n) for dataset of size {n:,}.")

    # --- Query parameter: probes ---
    probes = max(1, int(math.sqrt(lists)))
    explanations.append(f"probes base: sqrt(lists) = {probes}")

    if profile.has_filters:
        sel = profile.filter_selectivity
        if sel < 0.01:
            probes = min(probes * 10, lists)
            explanations.append(
                f"probes increased 10x for very selective filter ({sel:.1%}). "
                f"Set to {probes}."
            )
        elif sel < 0.05:
            probes = min(probes * 5, lists)
            explanations.append(
                f"probes increased 5x for selective filter ({sel:.1%}). "
                f"Set to {probes}."
            )
        elif sel < 0.1:
            probes = min(probes * 3, lists)
            explanations.append(
                f"probes tripled for moderate filter ({sel:.0%}). "
                f"Set to {probes}."
            )
        else:
            probes = min(probes * 2, lists)
            explanations.append(
                f"probes doubled for filtered workload ({sel:.0%}). "
                f"Set to {probes}."
            )

    build_params = {"lists": lists}
    query_params = {"probes": probes}

    return build_params, query_params, explanations


def recommend_auxiliary_indexes(profile: WorkloadProfile) -> list[dict]:
    """Recommend auxiliary B-tree indexes for filter columns.

    Args:
        profile: WorkloadProfile.

    Returns:
        List of dicts with column and index_type info.
    """
    auxiliary = []
    if profile.has_filters and profile.filter_columns:
        for col in profile.filter_columns:
            auxiliary.append({
                "column": col,
                "index_type": "btree",
                "reason": f"B-tree index on '{col}' to accelerate filtered vector queries.",
            })
    return auxiliary


def recommend_session_settings(
    profile: WorkloadProfile, index_type: str
) -> dict:
    """Recommend PostgreSQL session settings.

    Args:
        profile: WorkloadProfile.
        index_type: Selected index type.

    Returns:
        Dict of session parameter name -> value.
    """
    settings = {}

    # work_mem: increase for larger datasets
    if profile.n_vectors > 500_000:
        settings["work_mem"] = "256MB"
    elif profile.n_vectors > 100_000:
        settings["work_mem"] = "128MB"
    else:
        settings["work_mem"] = "64MB"

    # maintenance_work_mem: for index builds
    if profile.n_vectors > 1_000_000:
        settings["maintenance_work_mem"] = "1GB"
    elif profile.n_vectors > 100_000:
        settings["maintenance_work_mem"] = "512MB"
    else:
        settings["maintenance_work_mem"] = "256MB"

    return settings


def generate_recommendation(profile: WorkloadProfile) -> Recommendation:
    """Generate a complete index recommendation for the given workload.

    Args:
        profile: WorkloadProfile with workload characteristics.

    Returns:
        Recommendation with index type, parameters, and explanations.
    """
    index_type, type_reason = select_index_type(profile)
    explanations = [type_reason]

    if index_type == "none":
        return Recommendation(
            index_type="none",
            explanation=explanations,
        )

    if index_type == "hnsw":
        build_params, query_params, param_explanations = recommend_hnsw_params(profile)
    else:
        build_params, query_params, param_explanations = recommend_ivfflat_params(profile)

    explanations.extend(param_explanations)

    auxiliary = recommend_auxiliary_indexes(profile)
    session_settings = recommend_session_settings(profile, index_type)

    return Recommendation(
        index_type=index_type,
        build_params=build_params,
        query_params=query_params,
        auxiliary_indexes=auxiliary,
        session_settings=session_settings,
        explanation=explanations,
    )
