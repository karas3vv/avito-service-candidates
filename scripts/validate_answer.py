"""Check the exact submission format and every item identifier."""

from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parent.parent
answer_path = root / "answer.csv"
queries_path = root / "data" / "benchmark_queries.parquet"
items_path = root / "data" / "benchmark_items.parquet"

answer = pd.read_csv(answer_path, dtype=str, keep_default_na=False)
queries = pd.read_parquet(queries_path, columns=["query_id"])
items = pd.read_parquet(items_path, columns=["item_id"])

assert list(answer.columns) == ["query_id", "answer"]
assert len(answer) == len(queries)
assert answer.query_id.is_unique
assert set(answer.query_id) == set(queries.query_id)
assert answer.query_id.str.len().eq(16).all()

valid_items = set(items.item_id)
for query_id, response in answer.itertuples(index=False):
    ids = response.split()
    assert len(ids) <= 50, query_id
    assert len(ids) == len(set(ids)), query_id
    assert all(item_id in valid_items for item_id in ids), query_id

print(f"OK: {len(answer)} queries, valid item IDs, at most 50 per query")
