"""
Preprocessing and query-driven dataframe filtering module for AML System.
Polars-based fast filtering operations.
"""

from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import polars as pl


def parse_timestamp_column(df: pl.DataFrame, col_name: str = "Timestamp") -> pl.DataFrame:
    """
    Parses timestamp string column into datetime format.
    Supports formats like 'YYYY/MM/DD HH:MM:SS' or ISO strings.
    """
    if col_name not in df.columns:
        return df

    # Check if already datetime
    if df[col_name].dtype in [pl.Datetime, pl.Date]:
        return df

    try:
        return df.with_columns(
            pl.col(col_name).str.to_datetime("%Y/%m/%d %H:%M:%S", strict=False).alias("parsed_timestamp")
        )
    except Exception:
        # Fallback parsing
        return df.with_columns(
            pl.col(col_name).str.to_datetime(strict=False).alias("parsed_timestamp")
        )


def apply_time_filter(df: pl.DataFrame, days: int | None = None, start_date: str | None = None, end_date: str | None = None) -> pl.DataFrame:
    """
    Applies time window filtering to dataframe.
    """
    if "parsed_timestamp" not in df.columns:
        df = parse_timestamp_column(df)

    if "parsed_timestamp" not in df.columns or df["parsed_timestamp"].null_count() == len(df):
        return df

    valid_ts_df = df.filter(pl.col("parsed_timestamp").is_not_null())
    if len(valid_ts_df) == 0:
        return df

    max_dt = valid_ts_df["parsed_timestamp"].max()

    if days is not None:
        min_dt = max_dt - timedelta(days=days)
        return valid_ts_df.filter(pl.col("parsed_timestamp") >= min_dt)

    if start_date:
        s_dt = datetime.fromisoformat(start_date)
        valid_ts_df = valid_ts_df.filter(pl.col("parsed_timestamp") >= s_dt)

    if end_date:
        e_dt = datetime.fromisoformat(end_date)
        valid_ts_df = valid_ts_df.filter(pl.col("parsed_timestamp") <= e_dt)

    return valid_ts_df


def apply_entity_filter(df: pl.DataFrame, entity_ids: list[str]) -> pl.DataFrame:
    """
    Filters dataframe for transactions involving specified entity/account IDs (sender or receiver).
    """
    if not entity_ids:
        return df

    # Search in both Account (sender) and Account.1 (receiver)
    clean_ids = [str(e).strip() for e in entity_ids]
    
    # Also handle numeric account matching (e.g. 4521 -> ACC_4521)
    expanded_ids = list(clean_ids)
    for eid in clean_ids:
        if eid.isdigit():
            expanded_ids.append(f"ACC_{int(eid):04d}")
            expanded_ids.append(f"ACC_{eid}")

    return df.filter(
        pl.col("Account").is_in(expanded_ids) | pl.col("Account.1").is_in(expanded_ids)
    )


def apply_amount_filter(df: pl.DataFrame, min_amount: float | None = None, max_amount: float | None = None) -> pl.DataFrame:
    """
    Filters transactions based on amount bounds.
    """
    filtered = df
    if min_amount is not None:
        filtered = filtered.filter(pl.col("Amount Received") >= min_amount)
    if max_amount is not None:
        filtered = filtered.filter(pl.col("Amount Received") < max_amount)
    return filtered
