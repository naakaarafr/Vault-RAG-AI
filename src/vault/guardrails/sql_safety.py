import sqlglot
from sqlglot import exp

DEFAULT_ALLOWLISTED_TABLES = {"documents", "chunks", "metadata", "users", "audit_log"}
DEFAULT_ALLOWLISTED_COLUMNS = {
    "documents": {"doc_id", "title", "source_url", "allowed_roles", "file_path"},
    "chunks": {"chunk_id", "doc_id", "title", "page", "allowed_roles", "text"},
    "metadata": {"doc_id", "key", "value"},
    "users": {"user_id", "role"},
    "audit_log": {"timestamp", "role", "action"}
}

def validate_sql(
    sql_query: str,
    allowlisted_tables: set[str] | None = None,
    allowlisted_columns: dict[str, set[str]] | None = None,
    max_limit: int = 100
) -> tuple[bool, str]:
    """
    Validates SQL query safety:
    1. SELECT statement only.
    2. Tables must be in allowlisted_tables.
    3. Columns must be in allowlisted_columns for the corresponding tables (if columns are explicitly specified).
    4. Prohibits state mutation (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, EXEC, etc.).
    5. Returns (is_safe, reason).
    """
    if not sql_query or not sql_query.strip():
        return False, "Empty SQL query."

    tables_allow = allowlisted_tables if allowlisted_tables is not None else DEFAULT_ALLOWLISTED_TABLES
    columns_allow = allowlisted_columns if allowlisted_columns is not None else DEFAULT_ALLOWLISTED_COLUMNS

    try:
        parsed_expressions = sqlglot.parse(sql_query)
    except Exception as e:
        return False, f"SQL syntax error / parse failure: {str(e)}"

    if not parsed_expressions or len(parsed_expressions) > 1:
        return False, "Query must contain exactly one SQL statement."

    expression = parsed_expressions[0]

    if not isinstance(expression, exp.Select):
        return False, f"Forbidden statement type '{expression.key.upper()}'. Only SELECT statements are permitted."

    # Validate tables referenced
    referenced_tables = {table.name.lower() for table in expression.find_all(exp.Table)}
    for table_name in referenced_tables:
        if table_name not in tables_allow:
            return False, f"Table '{table_name}' is not in the security allowlist."

    # Validate column references
    for column in expression.find_all(exp.Column):
        col_name = column.name.lower()
        if col_name == "*":
            continue
        
        # If table name is explicitly qualified
        table_name = column.table.lower() if column.table else None
        if table_name and table_name in columns_allow:
            if col_name not in columns_allow[table_name]:
                return False, f"Column '{col_name}' on table '{table_name}' is not in the security allowlist."
        elif not table_name:
            # Check if column belongs to any allowlisted table
            found = any(col_name in allowed_cols for allowed_cols in columns_allow.values())
            if not found:
                return False, f"Column '{col_name}' is not in the security allowlist."

    return True, "SQL query is safe."
