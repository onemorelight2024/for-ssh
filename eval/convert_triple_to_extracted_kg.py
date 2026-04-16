#!/usr/bin/env python3
"""把 baseline 产物里的 triple 转成 evaluate_fact_entailment.py 需要的 extracted_kg。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def triples_to_extracted_kg(triples: Any) -> Dict[str, List]:
    entities: set[str] = set()
    edges: set[str] = set()
    relations: List[List[str]] = []
    seen: set[tuple[str, str, str]] = set()

    if not isinstance(triples, list):
        triples = []

    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        s, p, o = (str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip())
        if not s or not p or not o:
            continue
        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)
        relations.append([s, p, o])
        entities.add(s)
        entities.add(o)
        edges.add(p)

    return {
        "entities": sorted(entities),
        "edges": sorted(edges),
        "relations": relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从每条记录的 triple 生成 extracted_kg（Graph 结构），供事实蕴含评测使用。"
    )
    parser.add_argument("--input", type=Path, required=True, help="baseline 输出 JSON（数组）")
    parser.add_argument("--output", type=Path, required=True, help="写出路径")
    parser.add_argument(
        "--triple-key",
        type=str,
        default="triple",
        help="三元组列表所在字段名（与 run_l2_*_baseline 的 --output-key 一致）",
    )
    parser.add_argument(
        "--overwrite-kg",
        action="store_true",
        help="若已有 extracted_kg 仍覆盖；默认跳过已存在字段",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("输入 JSON 顶层必须是数组")

    for item in data:
        if not isinstance(item, dict):
            continue
        if "extracted_kg" in item and not args.overwrite_kg:
            continue
        item["extracted_kg"] = triples_to_extracted_kg(item.get(args.triple_key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
