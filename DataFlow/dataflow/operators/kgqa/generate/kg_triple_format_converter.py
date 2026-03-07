"""
KGTripleFormatConverter：将三元组转成 Numpy 或 NetworkX 格式的图存储。
支持两种三元组格式：
- 实体关系：<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by
- 属性：<entity> Henry <attribute> nationality <value> Canadian
针对pipeline，图数据保存为 graph.npz，供 PathSampler 从 kg_path 加载。而NetworkX作为run函数的返回值，如果其他pipeline需要使用该算子，供其他算子使用。
"""
import os          
import re         
import json        
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import networkx as nx  # 可选：output_format=networkx 时构建 nx.DiGraph


@OPERATOR_REGISTRY.register()
class KGTripleFormatConverter(OperatorABC):
    """算子：triple -> edges/node_labels/relation_labels/attr_* -> dataframe 列 + graph.npz"""

    def __init__(self, output_format: str = "numpy", save_npz: bool = True):
        self.logger = get_logger()
        self.output_format = output_format  
        self.save_npz = save_npz           

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "将三元组转成 Numpy、NetworkX 或 pickle 格式的图存储。"
        return "Convert triples to Numpy, NetworkX or pickle graph storage format."

    def _validate_dataframe(self, dataframe: pd.DataFrame, triple_key: str):
        """校验必须列 triple 存在"""
        if triple_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {triple_key}")

    def run(self, storage: DataFlowStorage = None):
        """读 dataframe -> 转换 -> 写回 + 保存 npz"""
        dataframe = storage.read("dataframe")
        triple_key = "triple"
        entity_key = "entity"
        self._validate_dataframe(dataframe, triple_key)

        processor = KGTripleFormatConverterProcessor(output_format=self.output_format)

        # 按行提取 triple 和 entity
        triples_list, entities_list = [], []
        for _, row in dataframe.iterrows():
            triples_list.append(
                KGTripleFormatConverterProcessor._normalize_triples_input(row.get(triple_key))
            )
            entities_list.append(
                row.get(entity_key) if entity_key in dataframe.columns else None
            )

        # 批量转换
        outputs = processor.convert_batch(triples_list, entities_list)
        output_keys = [
            "edges", "node_labels", "relation_labels",
            "attr_entity_ids", "attr_attribute_labels", "attr_values",
        ]
        # 将输出写回 dataframe 各列
        for k in output_keys:
            if k not in dataframe.columns:
                dataframe[k] = None
        for idx, output in zip(dataframe.index, outputs):
            if output:
                for k in output_keys:
                    dataframe.at[idx, k] = output.get(k)

        # networkx 格式时，构建 graph_list 作为返回值
        graph_list = None
        if self.output_format == "networkx":
            graph_list = [o.get("graph") if o else None for o in outputs]

        # 获取下一步缓存路径，用于确定 npz 保存目录
        output_path_fn = getattr(storage, "_get_cache_file_path", None)
        output_file = output_path_fn(storage.operator_step + 1) if callable(output_path_fn) else None

        # 保存 graph.npz 到 {base_dir}/{base_name}_kg/graph.npz
        if outputs and self.save_npz and output_file:
            base_dir = os.path.dirname(output_file)
            base_name = os.path.splitext(os.path.basename(output_file))[0]
            kg_root = os.path.join(base_dir, f"{base_name}_kg")
            os.makedirs(kg_root, exist_ok=True)
            first_output = next((o for o in outputs if o), None)
            if first_output:
                KGTripleFormatConverterProcessor._save_graph_npz(first_output, kg_root)
                self.logger.info(f"Graph saved to {os.path.join(kg_root, 'graph.npz')}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        if self.output_format == "networkx":
            return {"output_keys": output_keys, "graph_nx_list": graph_list}
        return {"output_keys": output_keys}


class KGTripleFormatConverterProcessor:
    """转换逻辑：解析三元组 -> 构建数组 -> 输出 dict；含 _save_graph_npz 供落盘"""

    def __init__(self, output_format: str = "numpy"):
        self.logger = get_logger()
        self.output_format = output_format

    @staticmethod
    def _save_graph_npz(output: Dict[str, Any], kg_root: str) -> None:
        """将 convert_batch 输出保存为 graph.npz。PathSampler 从 kg_path 加载此文件。"""
        edges = output.get("edges")
        node_labels = output.get("node_labels")
        relation_labels = output.get("relation_labels")
        attr_entity_ids = output.get("attr_entity_ids")
        attr_attribute_labels = output.get("attr_attribute_labels")
        attr_values = output.get("attr_values")
        if edges is None or node_labels is None or relation_labels is None:
            return
        # 生成 QID/PID 占位符，与 PathSampler 约定一致
        node_ids = np.array([f"Q{i:04d}" for i in range(len(node_labels))], dtype=object)
        relation_ids = np.array([f"P{i:04d}" for i in range(len(relation_labels))], dtype=object)
        npz_path = os.path.join(kg_root, "graph.npz")
        np.savez(
            npz_path,
            edges=edges,
            node_labels=node_labels,
            relation_labels=relation_labels,
            node_ids=node_ids,
            relation_ids=relation_ids,
            attr_entity_ids=attr_entity_ids if attr_entity_ids is not None else np.array([], dtype=np.int64),
            attr_attribute_labels=attr_attribute_labels if attr_attribute_labels is not None else np.array([], dtype=object),
            attr_values=attr_values if attr_values is not None else np.array([], dtype=object),
        )

    @staticmethod
    def _normalize_triples_input(triples) -> List:
        """统一输入格式：str(json/list) -> List[str]"""
        if isinstance(triples, str):
            try:
                return json.loads(triples)
            except Exception:
                return [t.strip() for t in triples.split("\n") if t.strip()]
        if isinstance(triples, list):
            if triples and isinstance(triples[0], dict):
                return [t.get("triple_str", "") for t in triples if isinstance(t, dict) and "triple_str" in t]
            return triples
        return []

    def convert_batch(self, triples_list: List[List[str]], entities_list: List[Optional[List[str]]]) -> List[Dict[str, Any]]:
        """逐行调用 _convert_triples"""
        results = []
        for triples, entities in zip(triples_list, entities_list):
            results.append(self._convert_triples(triples, entities))
        return results

    def _convert_triples(self, triples: List[str], entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """核心：解析三元组 -> 构建 node/relation 映射 -> 输出 edges 等数组"""
        parsed_er_triples: List[Tuple[str, str, str]] = []  # (subject, object, relation)
        parsed_attr_triples: List[Tuple[str, str, str]] = []  # (entity, attribute, value)
        all_entities = set()
        all_relations = set()

        for triple_str in triples or []:
            er_parsed = self._parse_entity_relation_triple(triple_str)
            if er_parsed:
                subject, obj, relation = er_parsed
                parsed_er_triples.append((subject, obj, relation))
                all_entities.add(subject)
                all_entities.add(obj)
                all_relations.add(relation)
                continue
            attr_parsed = self._parse_attribute_triple(triple_str)
            if attr_parsed:
                entity, attribute, value = attr_parsed
                parsed_attr_triples.append((entity, attribute, value))
                all_entities.add(entity)

        # 合并 entity 列中的额外实体
        if entities:
            _flat = []
            for e in ([entities] if isinstance(entities, str) else entities or []):
                if isinstance(e, str):
                    _flat.extend(x.strip() for x in e.split(","))
                elif isinstance(e, list):
                    _flat.extend(str(x).strip() for x in e)
            all_entities.update(_flat)

        node_list = sorted(list(all_entities))
        relation_list = sorted(list(all_relations))
        node_to_id = {node: idx for idx, node in enumerate(node_list)}
        relation_to_id = {rel: idx for idx, rel in enumerate(relation_list)}

        # 构建边：实体关系三元组 -> (head_id, tail_id), relation_type_id
        edge_list = []
        relation_type_list = []
        for subject, obj, relation in parsed_er_triples:
            if subject in node_to_id and obj in node_to_id and relation in relation_to_id:
                edge_list.append((node_to_id[subject], node_to_id[obj]))
                relation_type_list.append(relation_to_id[relation])

        # 构建属性三元组数组
        attr_entity_ids_list = []
        attr_attribute_labels_list = []
        attr_values_list = []
        for entity, attribute, value in parsed_attr_triples:
            if entity in node_to_id:
                attr_entity_ids_list.append(node_to_id[entity])
                attr_attribute_labels_list.append(attribute)
                attr_values_list.append(value)

        # edges: (3, E)，row0=head, row1=tail, row2=relation_type_id
        edges = (
            np.stack([
                np.array([e[0] for e in edge_list], dtype=np.int64),
                np.array([e[1] for e in edge_list], dtype=np.int64),
                np.array(relation_type_list, dtype=np.int64),
            ], axis=0)
            if edge_list else np.zeros((3, 0), dtype=np.int64)
        )
        node_labels = np.array(node_list, dtype=object)
        relation_labels = np.array(relation_list, dtype=object)
        attr_entity_ids = np.array(attr_entity_ids_list, dtype=np.int64)
        attr_attribute_labels = np.array(attr_attribute_labels_list, dtype=object)
        attr_values = np.array(attr_values_list, dtype=object)

        # 可选：构建 networkx 图
        graph = None
        if self.output_format == "networkx" and nx:
            graph = nx.DiGraph()
            for i, (subject, obj, relation) in enumerate(parsed_er_triples):
                if subject in node_to_id and obj in node_to_id:
                    graph.add_edge(subject, obj, relation=relation, edge_id=i)
            for node in node_list:
                if node not in graph:
                    graph.add_node(node)

        return {
            "edges": edges,
            "node_labels": node_labels,
            "relation_labels": relation_labels,
            "attr_entity_ids": attr_entity_ids,
            "attr_attribute_labels": attr_attribute_labels,
            "attr_values": attr_values,
            "graph": graph,
        }

    @staticmethod
    def _normalize_triple_str(triple_str: str) -> Optional[str]:
        """去除 <triplet>、末尾句号等"""
        if not triple_str or not isinstance(triple_str, str):
            return None
        s = triple_str.strip().replace("<triplet>", "").strip()
        return s[:-1].strip() if s.endswith(".") else s if s else None

    @staticmethod
    def _parse_triple_by_pattern(triple_str: str, markers: Tuple[str, str, str], pattern: str) -> Optional[Tuple[str, str, str]]:
        """通用：按 markers 和正则解析，返回 (a, b, c)"""
        s = KGTripleFormatConverterProcessor._normalize_triple_str(triple_str)
        if not s or not all(m in s for m in markers):
            return None
        try:
            match = re.search(pattern, s, re.DOTALL)
            if match:
                a, b, c = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                return (a, b, c) if (a and b and c) else None
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_entity_relation_triple(triple_str: str) -> Optional[Tuple[str, str, str]]:
        """解析 <subj> X <obj> Y <rel> R -> (X, Y, R)"""
        return KGTripleFormatConverterProcessor._parse_triple_by_pattern(
            triple_str, ("<subj>", "<obj>", "<rel>"), r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+)",
        )

    @staticmethod
    def _parse_attribute_triple(triple_str: str) -> Optional[Tuple[str, str, str]]:
        """解析 <entity> E <attribute> A <value> V -> (E, A, V)"""
        return KGTripleFormatConverterProcessor._parse_triple_by_pattern(
            triple_str, ("<entity>", "<attribute>", "<value>"), r"<entity>\s*(.+?)\s*<attribute>\s*(.+?)\s*<value>\s*(.+)",
        )
