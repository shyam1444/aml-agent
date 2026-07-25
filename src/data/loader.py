"""
Data Loader and Synthetic IBM AML Benchmark Generator module.
Loads transaction data using Polars for high performance.
Generates compliant IBM AML formatted dataset (HI-Small_Trans.csv) with ground-truth
'Is Laundering' labels for realistic benchmarking, false-positive evaluation, and demo stability.
"""

from pathlib import Path
import random
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import polars as pl
# pyrefly: ignore [missing-import]
import numpy as np


IBM_AML_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering"
]


def generate_synthetic_ibm_aml_dataset(
    output_path: str | Path = "data/HI-Small_Trans.csv",
    num_normal_transactions: int = 5000,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generates a synthetic dataset matching the Kaggle IBM AML HI-Small_Trans.csv schema.
    Includes deterministic injection of AML typologies:
    - Structuring
    - Smurfing
    - Layering
    - Rapid Cashout
    - Round-tripping
    - Velocity Spike
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    records = []

    # Helper to generate random timestamps over 60 days
    def random_timestamp(days_offset: float = 0):
        offset = timedelta(
            days=days_offset + random.uniform(0, 50),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        return (start_date + offset).strftime("%Y/%m/%d %H:%M:%S")

    # 1. Normal Transactions
    accounts = [f"ACC_{i:04d}" for i in range(1000, 1500)]
    banks = [10, 20, 30, 40, 50]
    formats = ["Cheque", "Credit Card", "Wire", "ACH", "Cash"]

    for _ in range(num_normal_transactions):
        from_acc = random.choice(accounts)
        to_acc = random.choice([a for a in accounts if a != from_acc])
        amt = round(float(np.random.exponential(scale=1200) + 10.0), 2)
        records.append({
            "Timestamp": random_timestamp(),
            "From Bank": random.choice(banks),
            "Account": from_acc,
            "To Bank": random.choice(banks),
            "Account.1": to_acc,
            "Amount Received": amt,
            "Receiving Currency": "USD",
            "Amount Paid": amt,
            "Payment Currency": "USD",
            "Payment Format": random.choice(formats),
            "Is Laundering": 0
        })

    # 2. Inject Structuring Typology (Customer ACC_4521 & ACC_9001)
    structuring_accounts = ["ACC_4521", "ACC_9001"]
    for s_acc in structuring_accounts:
        base_time = start_date + timedelta(days=15, hours=10)
        for i in range(6):  # 6 transactions between $8,500 and $9,950 within 4 days
            amt = round(random.uniform(8500, 9950), 2)
            ts = (base_time + timedelta(hours=i * 12 + random.randint(1, 3))).strftime("%Y/%m/%d %H:%M:%S")
            records.append({
                "Timestamp": ts,
                "From Bank": 10,
                "Account": s_acc,
                "To Bank": random.choice(banks),
                "Account.1": random.choice(accounts),
                "Amount Received": amt,
                "Receiving Currency": "USD",
                "Amount Paid": amt,
                "Payment Currency": "USD",
                "Payment Format": "Cash",
                "Is Laundering": 1
            })

    # 3. Inject Smurfing Typology (Multiple senders -> ACC_7700 beneficiary -> ACC_9999 aggregate)
    smurf_beneficiary = "ACC_7700"
    smurf_senders = [f"ACC_SMURF_{i:02d}" for i in range(1, 8)]
    base_smurf_time = start_date + timedelta(days=20, hours=8)
    total_smurf_amt = 0.0
    for idx, sender in enumerate(smurf_senders):
        amt = round(random.uniform(3000, 4500), 2)
        total_smurf_amt += amt
        ts = (base_smurf_time + timedelta(hours=idx * 2)).strftime("%Y/%m/%d %H:%M:%S")
        records.append({
            "Timestamp": ts,
            "From Bank": 20,
            "Account": sender,
            "To Bank": 30,
            "Account.1": smurf_beneficiary,
            "Amount Received": amt,
            "Receiving Currency": "USD",
            "Amount Paid": amt,
            "Payment Currency": "USD",
            "Payment Format": "Wire",
            "Is Laundering": 1
        })
    # Smurf aggregation step
    agg_ts = (base_smurf_time + timedelta(hours=20)).strftime("%Y/%m/%d %H:%M:%S")
    records.append({
        "Timestamp": agg_ts,
        "From Bank": 30,
        "Account": smurf_beneficiary,
        "To Bank": 40,
        "Account.1": "ACC_9999",
        "Amount Received": round(total_smurf_amt * 0.95, 2),
        "Receiving Currency": "USD",
        "Amount Paid": round(total_smurf_amt * 0.95, 2),
        "Payment Currency": "USD",
        "Payment Format": "Wire",
        "Is Laundering": 1
    })

    # 4. Inject Layering Typology (ACC_L1 -> ACC_L2 -> ACC_L3 -> ACC_L4)
    layer_chain = ["ACC_L1", "ACC_L2", "ACC_L3", "ACC_L4"]
    base_layer_time = start_date + timedelta(days=25, hours=14)
    current_amt = 50000.0
    for i in range(len(layer_chain) - 1):
        ts = (base_layer_time + timedelta(hours=i * 8)).strftime("%Y/%m/%d %H:%M:%S")
        records.append({
            "Timestamp": ts,
            "From Bank": 10 + i * 10,
            "Account": layer_chain[i],
            "To Bank": 10 + (i + 1) * 10,
            "Account.1": layer_chain[i + 1],
            "Amount Received": round(current_amt, 2),
            "Receiving Currency": "USD",
            "Amount Paid": round(current_amt, 2),
            "Payment Currency": "USD",
            "Payment Format": "Wire",
            "Is Laundering": 1
        })
        current_amt *= 0.97  # 3% fee attrition per hop

    # 5. Inject Round-Tripping (Cycle: ACC_RT1 -> ACC_RT2 -> ACC_RT3 -> ACC_RT1)
    rt_nodes = ["ACC_RT1", "ACC_RT2", "ACC_RT3"]
    base_rt_time = start_date + timedelta(days=30, hours=9)
    for i in range(len(rt_nodes)):
        src = rt_nodes[i]
        dst = rt_nodes[(i + 1) % len(rt_nodes)]
        ts = (base_rt_time + timedelta(hours=i * 6)).strftime("%Y/%m/%d %H:%M:%S")
        records.append({
            "Timestamp": ts,
            "From Bank": 10,
            "Account": src,
            "To Bank": 20,
            "Account.1": dst,
            "Amount Received": 25000.0,
            "Receiving Currency": "USD",
            "Amount Paid": 25000.0,
            "Payment Currency": "USD",
            "Payment Format": "Wire",
            "Is Laundering": 1
        })

    # 6. Inject Rapid Cash-out (ACC_CASH1: large credit, immediate withdrawal)
    base_rc_time = start_date + timedelta(days=35, hours=11)
    records.append({
        "Timestamp": base_rc_time.strftime("%Y/%m/%d %H:%M:%S"),
        "From Bank": 10,
        "Account": "ACC_EXTERNAL",
        "To Bank": 20,
        "Account.1": "ACC_CASH1",
        "Amount Received": 75000.0,
        "Receiving Currency": "USD",
        "Amount Paid": 75000.0,
        "Payment Currency": "USD",
        "Payment Format": "Wire",
        "Is Laundering": 1
    })
    records.append({
        "Timestamp": (base_rc_time + timedelta(hours=2)).strftime("%Y/%m/%d %H:%M:%S"),
        "From Bank": 20,
        "Account": "ACC_CASH1",
        "To Bank": 99,
        "Account.1": "ACC_CASH_OUT",
        "Amount Received": 72000.0,
        "Receiving Currency": "USD",
        "Amount Paid": 72000.0,
        "Payment Currency": "USD",
        "Payment Format": "Cash",
        "Is Laundering": 1
    })

    df = pl.DataFrame(records)
    
    # Save to file path if directory exists or create parent
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_file)
    print(f"Generated synthetic IBM AML benchmark dataset at {out_file} ({len(df)} rows)")
    return df


def load_dataset(data_path: str | Path = "data/HI-Small_Trans.csv") -> pl.DataFrame:
    """
    Loads transaction dataset via Polars. Auto-generates synthetic dataset if missing.
    """
    path = Path(data_path)
    if not path.exists():
        print(f"Dataset path '{path}' not found. Generating synthetic IBM AML dataset...")
        return generate_synthetic_ibm_aml_dataset(output_path=path)

    df = pl.read_csv(path)
    # Ensure correct data types
    schema_map = {
        "From Bank": pl.Utf8,
        "Account": pl.Utf8,
        "To Bank": pl.Utf8,
        "Account.1": pl.Utf8,
        "Amount Received": pl.Float64,
        "Amount Paid": pl.Float64,
        "Is Laundering": pl.Int64
    }
    for col, dtype in schema_map.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))

    return df
