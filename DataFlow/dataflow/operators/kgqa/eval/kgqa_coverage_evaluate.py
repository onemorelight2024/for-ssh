"""
KGQA 覆盖度评估算子：每个条目累加，求对总 KG 的覆盖度。
review过
"""
import pandas as pd
from typing import List, Optional, Dict, Any, Set
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class KGQACoverageEvaluate(OperatorABC):
    r"""
    KG 覆盖度评估：汇总所有条目的实体/关系，计算对总KG的覆盖比例。
    """

    def __init__(self):
        self.logger = get_logger()
        self.processor = KGQACoverageEvaluateProcessor()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "KG 覆盖度评估：汇总各条目的实体与关系，计算对总 KG 的覆盖比例。"
        return "KG coverage evaluation: aggregate entities/relations from entries, compute coverage ratio."

    def process_batch(
        self,
        path_infos: List[Dict[str, Any]],
        num_total_entities: Optional[int] = None,
        num_total_relations: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """批量计算覆盖度（整表一个分数）。"""
        result = self.processor.compute_coverage(path_infos, num_total_entities, num_total_relations)
        return [result]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if "path_info" not in dataframe.columns:
            raise ValueError("Missing required column: path_info")

    def run(self, storage: DataFlowStorage = None):
        r"""
        仅通过 storage.read("dataframe") 和 storage.write(dataframe) 读写。
        path_info、num_total_entities、num_total_relations 均从 dataframe 列读取。
        """
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        path_infos = dataframe["path_info"].tolist()
        path_infos = [p if isinstance(p, dict) else {} for p in path_infos]
        num_total_entities = dataframe["num_total_entities"].iloc[0] if "num_total_entities" in dataframe.columns else None
        num_total_relations = dataframe["num_total_relations"].iloc[0] if "num_total_relations" in dataframe.columns else None

        outputs = self.process_batch(path_infos, num_total_entities, num_total_relations)
        result = outputs[0] if outputs else {}

        dataframe["kgqa_entity_coverage"] = result.get("entity_coverage", 0.0)
        dataframe["kgqa_relation_coverage"] = result.get("relation_coverage", 0.0)

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [
            "kgqa_entity_coverage",
            "kgqa_relation_coverage",
        ]


class KGQACoverageEvaluateProcessor:
    """覆盖度计算逻辑。"""

    def __init__(self):
        self.logger = get_logger()

    def compute_coverage(
        self,
        path_infos: List[Dict[str, Any]],
        num_total_entities: Optional[int] = None,
        num_total_relations: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        汇总所有 path_info 中的实体和关系，计算覆盖度。
        覆盖度 = 覆盖数 / 总数（若总数为 0 则返回 0）
        """
        entities: Set[str] = set()
        relations: Set[str] = set()

        for pi in path_infos:
            if not isinstance(pi, dict):
                continue
            # 优先用 label 作为全局去重口径（qid/pid 在不同条目里可能从 0 重新编号）
            nodes = pi.get("path_nodes_label") or pi.get("path_nodes_qid") or []
            rels = pi.get("path_relations_label") or pi.get("path_relations_pid") or []
            for n in nodes:
                entities.add(str(n))
            for r in rels:
                relations.add(str(r))

        n_ent = len(entities)
        n_rel = len(relations)
        total_ent = num_total_entities if num_total_entities is not None and num_total_entities > 0 else 1
        total_rel = num_total_relations if num_total_relations is not None and num_total_relations > 0 else 1

        entity_coverage = n_ent / total_ent if total_ent > 0 else 0.0
        relation_coverage = n_rel / total_rel if total_rel > 0 else 0.0
        combined = (entity_coverage + relation_coverage) / 2.0

        return {
            "entity_coverage": entity_coverage,
            "relation_coverage": relation_coverage,
            "combined_coverage": combined,
            "num_covered_entities": n_ent,
            "num_covered_relations": n_rel,
        }
