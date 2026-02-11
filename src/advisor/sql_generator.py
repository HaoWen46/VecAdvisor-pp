"""SQL generation for VecAdvisor++ recommendations.

Produces concrete, executable SQL statements for index creation,
session settings, auxiliary indexes, and vector queries.
"""

from src.advisor.rules import Recommendation


def generate_create_index_sql(
    recommendation: Recommendation,
    table_name: str,
    column: str = "embedding",
) -> str | None:
    """Generate CREATE INDEX SQL for the recommended vector index.

    Args:
        recommendation: Advisor recommendation.
        table_name: Target table.
        column: Vector column name.

    Returns:
        SQL string or None if no index recommended.
    """
    if recommendation.index_type == "none":
        return None

    index_name = f"idx_{table_name}_{recommendation.index_type}"

    if recommendation.index_type == "hnsw":
        m = recommendation.build_params.get("m", 16)
        ef_construction = recommendation.build_params.get("ef_construction", 128)
        return (
            f"CREATE INDEX {index_name} ON {table_name} "
            f"USING hnsw ({column} vector_l2_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction});"
        )
    elif recommendation.index_type == "ivfflat":
        lists = recommendation.build_params.get("lists", 1000)
        return (
            f"CREATE INDEX {index_name} ON {table_name} "
            f"USING ivfflat ({column} vector_l2_ops) "
            f"WITH (lists = {lists});"
        )
    return None


def generate_session_settings_sql(recommendation: Recommendation) -> list[str]:
    """Generate SET statements for session-level parameters.

    Args:
        recommendation: Advisor recommendation.

    Returns:
        List of SET SQL statements.
    """
    statements = []

    # Index-specific query params
    if recommendation.index_type == "hnsw":
        ef_search = recommendation.query_params.get("ef_search", 40)
        statements.append(f"SET hnsw.ef_search = {ef_search};")
    elif recommendation.index_type == "ivfflat":
        probes = recommendation.query_params.get("probes", 10)
        statements.append(f"SET ivfflat.probes = {probes};")

    # General session settings
    for param, value in recommendation.session_settings.items():
        statements.append(f"SET {param} = '{value}';")

    return statements


def generate_auxiliary_index_sql(
    recommendation: Recommendation, table_name: str
) -> list[str]:
    """Generate CREATE INDEX SQL for auxiliary B-tree indexes.

    Args:
        recommendation: Advisor recommendation.
        table_name: Target table.

    Returns:
        List of CREATE INDEX SQL statements.
    """
    statements = []
    for aux in recommendation.auxiliary_indexes:
        col = aux["column"]
        idx_name = f"idx_{table_name}_{col}_btree"
        statements.append(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({col});"
        )
    return statements


def generate_query_sql(
    table_name: str,
    query_vector_placeholder: str = "%s",
    filter_clause: str | None = None,
    k: int = 10,
    column: str = "embedding",
) -> str:
    """Generate a vector similarity query SQL template.

    Args:
        table_name: Table to query.
        query_vector_placeholder: Placeholder for the query vector.
        filter_clause: Optional WHERE clause (without 'WHERE').
        k: Number of results.
        column: Vector column name.

    Returns:
        SQL query string.
    """
    where = f"WHERE {filter_clause}\n    " if filter_clause else ""
    return (
        f"SELECT id, {column} <-> {query_vector_placeholder}::vector AS distance\n"
        f"FROM {table_name}\n"
        f"    {where}"
        f"ORDER BY {column} <-> {query_vector_placeholder}::vector\n"
        f"LIMIT {k};"
    )


def generate_full_recommendation_sql(
    recommendation: Recommendation,
    table_name: str,
    column: str = "embedding",
) -> str:
    """Generate the complete set of SQL statements for a recommendation.

    Args:
        recommendation: Advisor recommendation.
        table_name: Target table.
        column: Vector column name.

    Returns:
        Complete SQL string with all statements.
    """
    lines = ["-- VecAdvisor++ Recommended Configuration", ""]

    # Session settings
    session_stmts = generate_session_settings_sql(recommendation)
    if session_stmts:
        lines.append("-- Session Settings")
        lines.extend(session_stmts)
        lines.append("")

    # Main vector index
    index_sql = generate_create_index_sql(recommendation, table_name, column)
    if index_sql:
        lines.append("-- Vector Index")
        lines.append(index_sql)
        lines.append("")
    else:
        lines.append("-- No vector index recommended (sequential scan sufficient)")
        lines.append("")

    # Auxiliary indexes
    aux_stmts = generate_auxiliary_index_sql(recommendation, table_name)
    if aux_stmts:
        lines.append("-- Auxiliary Indexes (for filter columns)")
        lines.extend(aux_stmts)
        lines.append("")

    # Example query
    lines.append("-- Example Query")
    lines.append(generate_query_sql(table_name, "'[...]'", None, 10, column))

    return "\n".join(lines)
