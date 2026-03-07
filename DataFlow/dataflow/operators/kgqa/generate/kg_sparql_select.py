"""
SparqlSelect：按模式名与 QA 数，从 SparqlLibrary 中按概率权重选择 SPARQL 模式生成 dataframe。

输入：storage 为空；模式名（simple/hard）与 QA 数显式传入
输出：与 KGSparqlPathSampler.jsonl 同结构的 dataframe，每行为一个按权重采样的 sparql_pattern。
"""
import json
import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class KGSparqlSelect(OperatorABC):
    r"""
    从 SparqlLibrary 按权重采样 SPARQL 模式，生成指定行数的 dataframe。
    支持简单模式（仅 1/2/3/10/11）与困难模式（全部，基础模式权重较低）。
    """

    SIMPLE_IDS = [1, 2, 3, 10, 11]
    SIMPLE_WEIGHTS = {
        1: 3,
        2: 4,
        3: 4,
        10: 2,
        11: 2,
    }
    HARD_BASE_IDS = [1, 10, 11]
    HARD_WEIGHTS = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 1,
        11: 1,
        12: 2,
    }

    def __init__(
        self,
        library_path: Optional[str] = None,
        mode: str = "simple",
        num_qa: int = 100,
        output_key: str = "sparql_pattern",
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            library_path: SparqlLibrary.json 路径，默认 DataFlow 目录下 SparqlLibrary.json
            mode: "simple" 简单模式（仅 id 1,2,3,10,11），"hard" 困难模式（全部，1/10/11 权重低）
            num_qa: 生成的 QA 行数
            output_key: 输出列名，用于兼容下游期望的 key
            seed: 随机种子，用于可复现
        """
        super().__init__(**kwargs)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        self.library_path = library_path or os.path.join(_root, "SparqlLibrary.json")
        self.mode = mode.strip().lower()
        self.num_qa = int(num_qa)
        self.output_key = output_key
        self.seed = seed

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "从 SparqlLibrary 按权重采样 SPARQL 模式，生成指定行数。"
        return "Sample SPARQL patterns from SparqlLibrary by weight, output N rows."

    def _load_library(self) -> List[Dict[str, Any]]:
        """加载 SparqlLibrary，返回 patterns 列表。"""
        if not os.path.isfile(self.library_path):
            raise FileNotFoundError(f"SparqlLibrary 不存在: {self.library_path}")
        with open(self.library_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        patterns = data.get("patterns", [])
        if not patterns:
            raise ValueError(f"SparqlLibrary 无有效 patterns: {self.library_path}")
        return patterns

    def _build_id_to_pattern(self, patterns: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return {int(p["id"]): p for p in patterns if "id" in p}

    def _get_weights_and_ids(self) -> tuple:
        if self.mode == "simple":
            ids = self.SIMPLE_IDS
            weights = [self.SIMPLE_WEIGHTS.get(i, 1) for i in ids]
        elif self.mode == "hard":
            ids = list(self.HARD_WEIGHTS.keys())
            weights = [self.HARD_WEIGHTS[i] for i in ids]
        else:
            raise ValueError(f"未知 mode: {self.mode}，支持 simple / hard")
        return ids, weights

    def run(self, storage: DataFlowStorage = None):
        r"""
        从 library 按权重采样 num_qa 个模式，组装 dataframe 并写入 storage。
        不依赖 storage 的输入数据，可直接作为 pipeline 首步。
        """
        if self.seed is not None:
            random.seed(self.seed)

        patterns = self._load_library()
        id_to_pattern = self._build_id_to_pattern(patterns)
        ids, weights = self._get_weights_and_ids()
        patterns_pool = [id_to_pattern[i] for i in ids if i in id_to_pattern]
        weights_pool = [weights[ids.index(i)] for i in ids if i in id_to_pattern]
        if not patterns_pool or not weights_pool:
            raise ValueError(f"mode={self.mode} 下无可用 pattern，ids={ids}")

        selected = random.choices(patterns_pool, weights=weights_pool, k=self.num_qa)
        rows = []
        for p in selected:
            row = {
                "id": p.get("id"),
                "sparql_des": p.get("sparql_des"),
                "sparql_pattern": p.get("sparql_pattern"),
                "sparql_content": p.get("sparql_content", p.get("sparql_pattern")),
                "graph_pattern": p.get("graph_pattern"),
                "entity_key": p.get("entity_key"),
            }
            if "path_sampler_compatible" in p:
                row["path_sampler_compatible"] = p["path_sampler_compatible"]
            rows.append(row)

        df = pd.DataFrame(rows)
        output_file = storage.write(df)
        self.logger.info(
            f"[SparqlSelect] mode={self.mode}, num_qa={self.num_qa}, "
            f"输出 {len(df)} 行 -> {output_file}"
        )
        return [self.output_key]
