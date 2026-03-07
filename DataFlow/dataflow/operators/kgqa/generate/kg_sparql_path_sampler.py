"""
KGSparqlPathSampler：根据 sparql_pattern 中的 op 操作在图上采样路径。

graph_pattern 格式：[[op_name, param1, param2, ...], ...]

【算子列表】
- sample_random: [entity_role] 从连通节点随机采 1 个实体，赋给 entity_role
- sample_relation: [relation_role, entity_role] relation 已绑定，采 1 个以该 relation 为出边的 subject，赋给 entity_role
- sample_entity_list: [entity_role] 从 entity_list 随机采 1 个，赋给 entity_role（entity_list 由 pipeline 注入）
- forward_find: [subject_role, relation_role, object_role] subject 输入，随机选 subject 的一个出边 relation，返回所有 object
- forward_find_attr: [subject_role, attr_role, value_role] subject 输入，随机选一个属性，返回所有 value
- reverse_find: [subject_role, relation_role, object_role] object 输入，随机选一个入边 relation，返回所有 subject
- sample_by_relation: [subject_role, relation_role, object_role] relation 输入，采 1 个 subject，返回所有 object
- find_relations_between: [entity1, entity2, relation] 或 [entity1, entity2, relation2, exclude_relation] 两端输入，后者排除 exclude_relation 后取 relation2
- or: [role1, role2, output_role] 取 role1 与 role2 的并集 list，赋给 output_role
- list_forward_find: [subject_role, relation_role, object_role] subject 为 list 时，对每个 entity 做 forward_find，object 输出为并集 list；SPARQL 填充用 <item1,item2,...> 格式
- sample_random_value: [value_role] 从图中属性值随机采 1 个，赋给 role_to_value，供 reverse_find_attr 等使用
- reverse_find_attr: [entity_role, property_role, value_role] 仅 value 输入，反查得 (property, entity)，一般唯一
- sample_val_gt: [entity_role, property_role, value_role] value 为阈值，采 1 个 value 更大的 entity
- sample_val_lt: [entity_role, property_role, value_role] value 为阈值，采 1 个 value 更小的 entity

图对象需具有 connected_nodes、neighbourhood_dict、edge_ids、relation_types、
node_labels、relation_labels、attr_entity_ids、attr_attribute_labels、attr_values 等属性。
图从 kg_path 目录下的 graph.npz 加载（由 TripleFormatConverter 保存）。
"""
import os.path as osp
import re
import json
import random
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.kgqa import SparqlCandidateSelectionPrompt


@prompt_restrict(SparqlCandidateSelectionPrompt)
@OPERATOR_REGISTRY.register()
class KGSparqlPathSampler(OperatorABC):
    r"""
    根据 sparql_pattern（SparqlLibrary 风格字典）中的原子操作在图上采样路径。
    图对象从 kg_path 目录的 graph.npz 加载（TripleFormatConverter 第一步保存）。
    """

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        kg_path: Optional[str] = None,
        num_candidates: Optional[int] = 3,
        **kwargs,
    ):
        """初始化：保存 llm_serving（多候选时用于 LLM 选择）、kg_path（npz的图路径）、num_candidates（候选数量）。"""
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.kg_path = kg_path
        self.num_candidates = num_candidates

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "根据 sparql_pattern 中的原子操作在图上采样路径。"
        return "Sample paths on the graph according to atomic ops in sparql_pattern."

    def process_batch(
        self,
        sparql_patterns: List[Dict[str, Any]],
        graph: object,
        node_qids: Optional[Any] = None,
        relation_pids: Optional[Any] = None,
        num_candidates: int = 1,
    ) -> List[Dict[str, Any]]:
        """批量采样入口：创建 Processor，对其调用 sample_batch。"""
        processor = KGSparqlPathSamplerProcessor(
            graph=graph,
            node_qids=node_qids,
            relation_pids=relation_pids,
            llm_serving=self.llm_serving,
            num_candidates=num_candidates,
        )
        return processor.sample_batch(sparql_patterns)

    def _validate_dataframe(self, dataframe: pd.DataFrame, input_key: str):
        """校验 dataframe 是否包含 input_key 列，缺失则抛错。"""
        if input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {input_key}")

    def run(self, storage: DataFlowStorage = None):
        """Pipeline 入口：从 storage 读 dataframe，从 kg_path 加载图，批量采样后写 path_info 到 output_key 列。"""
        dataframe = storage.read("dataframe")
        input_key = "sparql_pattern"
        output_key = "path_info"
        self._validate_dataframe(dataframe, input_key)

        #加载KG
        if not self.kg_path:
            raise ValueError("kg_path 未设置。请先运行 TripleFormatConverter 生成 graph.npz。")
        graph, node_qids, relation_pids = (
            KGSparqlPathSamplerProcessor.load_kg_graphs(self.kg_path)
        )
        node_labels = KGSparqlPathSamplerProcessor._graph_get(graph, "node_labels", [])
        relation_labels = KGSparqlPathSamplerProcessor._graph_get(graph, "relation_labels", [])
        num_total_entities = len(node_labels) if node_labels is not None else 0
        num_total_relations = len(relation_labels) if relation_labels is not None else 0
        try:
            self.logger.info(
                f"[PathSampler] graph loaded from {self.kg_path}, "
                f"nodes={len(KGSparqlPathSamplerProcessor._graph_get(graph, 'node_labels', []))}, "
                f"edges={KGSparqlPathSamplerProcessor._graph_get(graph, 'edge_ids', np.empty((2, 0))).shape[1]}, "
                f"relations={len(KGSparqlPathSamplerProcessor._graph_get(graph, 'relation_labels', []))}, "
                f"connected={len(KGSparqlPathSamplerProcessor._graph_get(graph, 'connected_nodes', []))}"
            )
        except Exception:
            self.logger.info("[PathSampler] graph loaded (stats unavailable)")

        #构造sparql_patterns列表供使用，保证每一行都有graph_pattern和entity_key
        sparql_patterns = []
        for _, row in dataframe.iterrows():
            v = row.get(input_key)
            if isinstance(v, dict):
                rec = dict(v)
                if "graph_pattern" in dataframe.columns and "graph_pattern" not in rec:
                    rec["graph_pattern"] = row.get("graph_pattern")
                if "entity_key" in dataframe.columns and "entity_key" not in rec:
                    rec["entity_key"] = row.get("entity_key")
                sparql_patterns.append(rec)
            else:
                rec = {"sparql_pattern": str(v) if v else None}
                if "graph_pattern" in dataframe.columns:
                    rec["graph_pattern"] = row.get("graph_pattern")
                if "entity_key" in dataframe.columns:
                    rec["entity_key"] = row.get("entity_key")
                sparql_patterns.append(rec)
        if sparql_patterns:
            sample_gp = sparql_patterns[0].get("graph_pattern")
            self.logger.info(
                f"[PathSampler] input_key={input_key}, rows={len(sparql_patterns)}, "
                f"first_has_graph_pattern={sample_gp is not None}"
            )

        num_candidates = int(self.num_candidates or 3)

        #调用process_batch进行采样
        outputs = self.process_batch(
            sparql_patterns,
            graph=graph,
            node_qids=node_qids,
            relation_pids=relation_pids,
            num_candidates=num_candidates,
        )

        #output相关
        if output_key not in dataframe.columns:
            dataframe[output_key] = None
        for idx, out in zip(dataframe.index, outputs):
            dataframe.at[idx, output_key] = out.get("path_info") if out else None

        dataframe["answer"] = [
            out.get("path_info", {}).get("answer") if out else None for out in outputs
        ]
        dataframe["num_total_entities"] = num_total_entities
        dataframe["num_total_relations"] = num_total_relations

        keep_cols = [
            c
            for c in [
                "sparql_des",
                "sparql_pattern",
                "path_info",
                "answer",
                "num_total_entities",
                "num_total_relations",
            ]
            if c in dataframe.columns
        ]
        if keep_cols:
            dataframe = dataframe[keep_cols]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]


class KGSparqlPathSamplerProcessor:
    r"""
    从 sparql_pattern 提取原子操作，在图上顺序执行，生成带 QID/语义名的 SPARQL 及三元组列表。
    graph_pattern 为 [op_name, param1, param2, ...] 的 op 列表。
    """

    def __init__(
        self,
        graph: object,
        node_qids=None,
        relation_pids=None,
        llm_serving=None,
        num_candidates=None,
        prompt_template=None,
        use_llm: bool = True,
        max_attempts: int = 100,
        **kwargs,
    ):
        """初始化 Processor：保存图、node/relation 映射、LLM、候选数等；索引 _attr_by_node 等懒加载。"""
        self.logger = get_logger()
        self.graph = graph
        self.node_qids = node_qids
        self.relation_pids = relation_pids
        self.rng = random.Random(None)
        self.llm_serving = llm_serving
        self.num_candidates = int(num_candidates or 1)
        self.prompt_template = prompt_template or SparqlCandidateSelectionPrompt()
        self.use_llm = bool(use_llm)
        self.max_attempts = int(max_attempts or 100)

    @staticmethod
    def _graph_get(g, key: str, default=None):
        """从图对象获取属性，支持 dict 与 object。"""
        if isinstance(g, dict):
            return g.get(key, default)
        return getattr(g, key, default)

    @staticmethod
    def load_kg_graphs(
        kg_path: str,
        entity_uri: str = "http://example.org/entities/",
        relation_uri: str = "http://example.org/relations/",
        exclude_rels: Optional[list] = None,
        exclude_nodes: Optional[np.ndarray] = None,
    ):
        """从 kg_path 目录加载 graph.npz（由 TripleFormatConverter 保存的）。返回 (graph_dict, node_qids, relation_pids)。"""
        exclude_rels = exclude_rels or []
        exclude_nodes = exclude_nodes if exclude_nodes is not None else np.array([], dtype=np.int32)

        npz_path = osp.join(kg_path, "graph.npz") if not kg_path.endswith(".npz") else kg_path
        if not osp.isfile(npz_path):
            raise FileNotFoundError(
                f"未找到 graph.npz：{npz_path}。请先运行 TripleFormatConverter 生成 graph.npz。"
            )

        graph, node_qids, relation_pids = KGSparqlPathSamplerProcessor._load_graph_from_npz(
            npz_path=npz_path,
            exclude_rels=exclude_rels,
            exclude_nodes=exclude_nodes,
        )
        return graph, node_qids, relation_pids

    @staticmethod
    def _load_graph_from_npz(
        *,
        npz_path: str,
        exclude_rels: Optional[List[int]] = None,
        exclude_nodes: Optional[List[int]] = None,
    ):
        """从 graph.npz 加载数组并构建图 dict。返回 (graph_dict, node_qids, relation_pids)。"""
        data = np.load(npz_path, allow_pickle=True)
        edges = data["edges"]
        node_labels = data["node_labels"]
        relation_labels = data["relation_labels"]
        node_qids = data["node_ids"] if "node_ids" in data else np.array(
            [f"Q{i:04d}" for i in range(len(node_labels))], dtype=object
        )
        relation_pids = data["relation_ids"] if "relation_ids" in data else np.array(
            [f"P{i:04d}" for i in range(len(relation_labels))], dtype=object
        )
        attr_entity_ids = data["attr_entity_ids"] if "attr_entity_ids" in data else None
        attr_attribute_labels = data["attr_attribute_labels"] if "attr_attribute_labels" in data else None
        attr_values = data["attr_values"] if "attr_values" in data else None

        graph = KGSparqlPathSamplerProcessor._build_graph_from_arrays(
            edges=edges,
            node_labels=node_labels,
            relation_labels=relation_labels,
            node_qids=node_qids,
            relation_pids=relation_pids,
            exclude_rels=exclude_rels or [],
            exclude_nodes=exclude_nodes or [],
            edge_callback=None,
            attr_entity_ids=attr_entity_ids,
            attr_attribute_labels=attr_attribute_labels,
            attr_values=attr_values,
        )
        return graph, node_qids, relation_pids

    @staticmethod
    def _build_graph_from_arrays(
        *,
        edges: Optional[np.ndarray] = None,
        edge_ids: Optional[np.ndarray] = None,
        relation_types: Optional[np.ndarray] = None,
        node_labels: Optional[np.ndarray] = None,
        relation_labels: Optional[np.ndarray] = None,
        node_qids: Optional[np.ndarray] = None,
        relation_pids: Optional[np.ndarray] = None,
        exclude_rels: Optional[List[int]] = None,
        exclude_nodes: Optional[List[int]] = None,
        edge_callback: Optional[Any] = None,
        attr_entity_ids: Optional[np.ndarray] = None,
        attr_attribute_labels: Optional[np.ndarray] = None,
        attr_values: Optional[np.ndarray] = None,
    ):
        """从边/节点/关系数组构建图 dict，含 neighbourhood_dict、connected_nodes 等，供采样使用。"""
        exclude_rels = exclude_rels if exclude_rels is not None else []
        exclude_nodes = exclude_nodes if exclude_nodes is not None else []
        if edges is not None:
            edge_ids = edges[:2]
            relation_types = edges[2]
        if edge_ids is None or relation_types is None:
            raise ValueError("需提供 (edge_ids, relation_types) 或 edges")

        msk = ~np.isin(edge_ids, np.array(exclude_nodes)).any(0)
        edge_ids = edge_ids[:, msk].astype(np.int64)
        relation_types = relation_types[msk].astype(np.int64)

        if node_qids is not None:
            node_labels = np.array(
                [f"{lbl} ({qid})" for lbl, qid in zip(node_labels, node_qids)],
                dtype=object,
            )
        else:
            node_labels = np.asarray(node_labels, dtype=object)

        if relation_pids is not None:
            relation_labels = np.array(
                [f"{lbl} ({pid})" for lbl, pid in zip(relation_labels, relation_pids)],
                dtype=object,
            )
        else:
            relation_labels = np.asarray(relation_labels, dtype=object)

        attr_entity_ids = (
            attr_entity_ids if attr_entity_ids is not None else np.array([], dtype=np.int64)
        )
        attr_attribute_labels = (
            attr_attribute_labels
            if attr_attribute_labels is not None
            else np.array([], dtype=object)
        )
        attr_values = (
            attr_values if attr_values is not None else np.array([], dtype=object)
        )

        neighbourhood_dict = KGSparqlPathSamplerProcessor._build_neighbourhood_dict(
            edge_ids=edge_ids,
            relation_types=relation_types,
            num_nodes=len(node_labels),
            exclude_rels=exclude_rels,
            edge_callback=edge_callback,
        )
        degree = KGSparqlPathSamplerProcessor._get_degree_array(neighbourhood_dict, len(node_labels))
        connected_nodes = np.where(degree > 0)[0]

        return {
            "edge_ids": edge_ids,
            "relation_types": relation_types,
            "exclude_rels": exclude_rels,
            "node_labels": node_labels,
            "relation_labels": relation_labels,
            "attr_entity_ids": attr_entity_ids,
            "attr_attribute_labels": attr_attribute_labels,
            "attr_values": attr_values,
            "neighbourhood_dict": neighbourhood_dict,
            "connected_nodes": connected_nodes,
        }

    @staticmethod
    def _build_neighbourhood_dict(
        *,
        edge_ids: np.ndarray,
        relation_types: np.ndarray,
        num_nodes: int,
        exclude_rels: Optional[List[int]] = None,
        edge_callback: Optional[Any] = None,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """构建邻接表：node -> {neigh_node -> [edge_indices]}，用于快速查找某节点的出边及邻接点。"""
        exclude_rels = exclude_rels if exclude_rels is not None else []
        nd: Dict[int, Dict[int, np.ndarray]] = {node: {} for node in range(num_nodes)}
        for edge_idx, (edge, rel) in enumerate(zip(edge_ids.T, relation_types)):
            if rel not in exclude_rels:
                h, t = int(edge[0]), int(edge[1])
                nd[h][t] = np.append(nd[h].get(t, np.array([], dtype=np.int64)), edge_idx)
                if edge_callback is not None:
                    try:
                        edge_callback(edge_idx, (h, t), int(rel))
                    except Exception:
                        pass
        return nd

    @staticmethod
    def _get_degree_array(
        neighbourhood_dict: Dict[int, Dict[int, np.ndarray]],
        num_nodes: int,
    ) -> np.ndarray:
        """计算每个节点的出度（邻居数），用于筛选 connected_nodes。"""
        degree = np.zeros(num_nodes, dtype=np.int64)
        for node, neighs in neighbourhood_dict.items():
            degree[node] = len(neighs)
        return degree

    def _get_attr_by_node(self) -> Dict[int, List[Tuple[int, str, Any]]]:
        """按节点聚合属性边，供 forward_find_attr、sample_attr_from_entity 等使用。node_idx -> [(attr_idx, attr_label, value), ...]。"""
        out = self._graph_get(self.graph, "attr_by_node", None)
        if out is not None:
            return out
        attr_e = self._graph_get(self.graph, "attr_entity_ids", None)
        attr_a = self._graph_get(self.graph, "attr_attribute_labels", None)
        attr_v = self._graph_get(self.graph, "attr_values", None)
        out: Dict[int, List[Tuple[int, str, Any]]] = {}
        if attr_e is not None and attr_a is not None and attr_v is not None:
            n = len(attr_e)
            for i in range(n):
                eid = int(attr_e[i])
                out.setdefault(eid, []).append(
                    (i, str(attr_a[i]).strip(), attr_v[i])
                )
        if isinstance(self.graph, dict):
            self.graph["attr_by_node"] = out
        else:
            setattr(self.graph, "attr_by_node", out)
        return out

    def _build_value_to_entities(self) -> Dict[Tuple[str, Any], List[int]]:
        """构建 (attr_label, value) -> [entity_ids] 反查索引。"""
        out = self._graph_get(self.graph, "value_to_entities", None)
        if out is not None:
            return out
        attr_e = self._graph_get(self.graph, "attr_entity_ids", None)
        attr_a = self._graph_get(self.graph, "attr_attribute_labels", None)
        attr_v = self._graph_get(self.graph, "attr_values", None)
        out: Dict[Tuple[str, Any], List[int]] = {}
        if attr_e is not None and attr_a is not None and attr_v is not None:
            for i in range(len(attr_e)):
                eid = int(attr_e[i])
                albl = str(attr_a[i]).strip()
                val = attr_v[i]
                key = (albl, str(val))
                out.setdefault(key, []).append(eid)
        if isinstance(self.graph, dict):
            self.graph["value_to_entities"] = out
        else:
            setattr(self.graph, "value_to_entities", out)
        return out

    def _build_value_to_property_entities(self) -> Dict[str, List[Tuple[str, int]]]:
        """构建 value_str -> [(property, entity_id), ...]，供 reverse_find_attr、sample_val_gt/lt 按 value 反查。"""
        out = self._graph_get(self.graph, "value_to_property_entities", None)
        if out is not None:
            return out
        attr_e = self._graph_get(self.graph, "attr_entity_ids", None)
        attr_a = self._graph_get(self.graph, "attr_attribute_labels", None)
        attr_v = self._graph_get(self.graph, "attr_values", None)
        out: Dict[str, List[Tuple[str, int]]] = {}
        if attr_e is not None and attr_a is not None and attr_v is not None:
            for i in range(len(attr_e)):
                eid = int(attr_e[i])
                albl = str(attr_a[i]).strip()
                val = attr_v[i]
                key = str(val)
                out.setdefault(key, []).append((albl, eid))
        if isinstance(self.graph, dict):
            self.graph["value_to_property_entities"] = out
        else:
            setattr(self.graph, "value_to_property_entities", out)
        return out

    def _build_property_values_sorted(self) -> Dict[str, List[Tuple[Any, int]]]:
        """构建 attr_label -> [(value, entity_id), ...] 并按 value 排序，供 sample_val_gt/lt 数值比较。"""
        out = self._graph_get(self.graph, "property_values_sorted", None)
        if out is not None:
            return out
        attr_e = self._graph_get(self.graph, "attr_entity_ids", None)
        attr_a = self._graph_get(self.graph, "attr_attribute_labels", None)
        attr_v = self._graph_get(self.graph, "attr_values", None)
        by_attr: Dict[str, List[Tuple[Any, int]]] = {}
        if attr_e is not None and attr_a is not None and attr_v is not None:
            for i in range(len(attr_e)):
                eid = int(attr_e[i])
                albl = str(attr_a[i]).strip()
                val = attr_v[i]
                by_attr.setdefault(albl, []).append((val, eid))
        for albl in list(by_attr.keys()):
            try:
                by_attr[albl] = sorted(by_attr[albl], key=lambda x: (x[0] is not None, x[0]))
            except TypeError:
                pass
        if isinstance(self.graph, dict):
            self.graph["property_values_sorted"] = by_attr
        else:
            setattr(self.graph, "property_values_sorted", by_attr)
        return by_attr

    @staticmethod
    def _parse_ops(gp: List[Any]) -> List[List[Any]]:
        """解析 graph_pattern 为 op 列表，每项 [op_name, param1, param2, ...]，统一转为 str。"""
        out: List[List[Any]] = []
        for item in gp:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                out.append([str(item[0])] + [str(x) for x in item[1:]])
        return out

    def _compute_op_order(
        self,
        ops: List[List[Any]],
    ) -> List[int]:
        """按依赖拓扑排序 op：B 依赖 A 的输出时 B 在 A 之后。返回 op 下标排列。"""
        n = len(ops)
        out_roles: Dict[int, Set[str]] = {}
        in_roles: Dict[int, Set[str]] = {}

        for i, op in enumerate(ops):
            op_name = op[0] if op else ""
            params = op[1:] if len(op) > 1 else []
            out_roles[i] = set()
            in_roles[i] = set()

            if op_name == "sample_random":
                if len(params) >= 1:
                    out_roles[i].add(params[0])
            elif op_name == "sample_relation":
                if len(params) >= 2:
                    in_roles[i].add(params[0])
                    out_roles[i].add(params[1])
            elif op_name == "sample_entity_list":
                if len(params) >= 1:
                    out_roles[i].add(params[0])
            elif op_name == "sample_random_value":
                if len(params) >= 1:
                    out_roles[i].add(params[0])
            elif op_name == "or":
                if len(params) >= 3:
                    in_roles[i].add(params[0])
                    in_roles[i].add(params[1])
                    out_roles[i].add(params[2])
            elif op_name == "list_forward_find":
                if len(params) >= 3:
                    in_roles[i].add(params[0])
                    in_roles[i].add(params[1])
                    out_roles[i].add(params[2])
            elif op_name in (
                "forward_find", "forward_find_attr", "reverse_find", "sample_by_relation",
                "find_relations_between", "reverse_find_attr", "sample_val_gt", "sample_val_lt",
            ):
                in_roles[i], out_roles[i] = self._op_io_roles(op_name, params)

        for i in range(n):
            op_name = ops[i][0] if ops[i] else ""
            params = ops[i][1:] if len(ops[i]) > 1 else []
            if op_name in ("forward_find", "reverse_find", "list_forward_find") and len(params) >= 2:
                r_tag = params[1]
                for j in range(i):
                    if r_tag in out_roles.get(j, set()):
                        in_roles[i].add(r_tag)
                        break

        order: List[int] = []
        remaining = set(range(n))

        def ready(idx: int) -> bool:
            for r in in_roles[idx]:
                if r not in _all_out:
                    return False
            return True

        _all_out: Set[str] = set()
        while remaining:
            found = False
            for idx in list(remaining):
                if ready(idx):
                    order.append(idx)
                    remaining.discard(idx)
                    _all_out.update(out_roles[idx])
                    found = True
                    break
            if not found:
                order.extend(remaining)
                break
        return order

    def _op_io_roles(self, op_name: str, params: List[str]) -> Tuple[Set[str], Set[str]]:
        """根据 op 类型和 params 推断 (input_roles, output_roles)，供 _compute_op_order 依赖分析。"""
        ins: Set[str] = set()
        outs: Set[str] = set()
        if op_name == "forward_find":
            if len(params) >= 3:
                ins.add(params[0])
                outs.add(params[1])
                outs.add(params[2])
        elif op_name == "forward_find_attr":
            if len(params) >= 3:
                ins.add(params[0])
                outs.add(params[1])
                outs.add(params[2])
        elif op_name == "reverse_find":
            if len(params) >= 3:
                ins.add(params[2])
                outs.add(params[0])
                outs.add(params[1])
        elif op_name == "sample_by_relation":
            if len(params) >= 3:
                ins.add(params[1])
                outs.add(params[0])
                outs.add(params[2])
        elif op_name == "find_relations_between":
            if len(params) >= 3:
                if len(params) == 3:
                    ins.add(params[0])
                    ins.add(params[2])
                    outs.add(params[1])
                else:
                    ins.add(params[0])
                    ins.add(params[1])
                    for i in range(2, len(params) - 1):
                        outs.add(params[i])
                    ins.add(params[-1])
        elif op_name == "reverse_find_attr":
            if len(params) >= 3:
                ins.add(params[2])
                outs.add(params[0])
                outs.add(params[1])
        elif op_name in ("sample_val_gt", "sample_val_lt"):
            if len(params) >= 3:
                ins.add(params[2])
                outs.add(params[0])
                outs.add(params[1])
        elif op_name == "list_forward_find":
            if len(params) >= 3:
                ins.add(params[0])
                ins.add(params[1])
                outs.add(params[2])
        return ins, outs

    def _rel_pid_to_idx(self, pid: str) -> Optional[int]:
        """将 relation pid 字符串映射为 relation_types 中的索引。需 relation_pids 已设置（由 load_kg_graphs 从 node_ids.npy/relation_ids.npy 加载或生成默认值）。"""
        if self.relation_pids is None:
            return None
        rp = np.asarray(self.relation_pids)
        for i in range(len(rp)):
            if str(rp[i]) == str(pid):
                return int(i)
        return None

    def _execute_op(
        self,
        op: List[Any],
        role_to_node: Dict[str, Union[int, List[int]]],
        role_to_value: Dict[str, str],
        edge_tag_to_pid: Dict[str, str],
        edge_tag_to_label: Dict[str, str],
        path_triples: List[Dict[str, Any]],
        exclude_entities: Set[int],
        entity_list: Optional[List[int]] = None,
    ) -> bool:
        """执行单个 op，原地更新 role_to_node、role_to_value、path_triples、edge_tag_*。exclude_entities 为采样时需排除的实体。成功返回 True，失败 False。"""
        if not op:
            return False
        op_name = str(op[0])
        params = [str(x) for x in op[1:]]

        attr_by_node = self._get_attr_by_node()
        nd = self._graph_get(self.graph, "neighbourhood_dict", {})
        edge_ids = self._graph_get(self.graph, "edge_ids", np.empty((2, 0)))
        rel_types = self._graph_get(self.graph, "relation_types", np.array([], dtype=np.int64))
        conn = self._graph_get(self.graph, "connected_nodes", np.array([], dtype=np.int64))

        def _pick_conn(exclude: Set[int]) -> Optional[int]:
            candidates = [int(c) for c in conn if int(c) not in exclude]
            return int(self.rng.choice(candidates)) if candidates else None

        def _add_er_triple(h: int, t: int, r_idx: int, edge_tag: str):
            r_pid = self._rel_idx_to_pid(r_idx)
            r_lbl = self._rel_idx_to_label(r_idx)
            edge_tag_to_pid[edge_tag] = r_pid
            edge_tag_to_label[edge_tag] = r_lbl
            path_triples.append({
                "head": self._node_to_qid(h),
                "relation": r_pid,
                "tail": self._node_to_qid(t),
                "triple_type": "entity_relation",
                "triple_str": f"<subj> {self._node_to_label(h)} <obj> {self._node_to_label(t)} <rel> {r_lbl}",
                "triple_str_readable": f"{self._node_to_label(h)} --{r_lbl}--> {self._node_to_label(t)}",
            })

        def _add_attr_triple(h: int, attr_label: str, val: Any, edge_tag: str):
            edge_tag_to_pid[edge_tag] = attr_label
            edge_tag_to_label[edge_tag] = attr_label
            path_triples.append({
                "head": self._node_to_qid(h),
                "relation": attr_label,
                "tail": str(val),
                "triple_type": "attribute",
                "triple_str": f"<subj> {self._node_to_label(h)} <obj> {val} <rel> {attr_label}",
                "triple_str_readable": f"{self._node_to_label(h)} --{attr_label}--> {val}",
            })

        def _safe_cmp(a: Any, b: Any, op: str) -> bool:
            try:
                fa, fb = float(a), float(b)
                return (fa > fb) if op == "gt" else (fa < fb)
            except (TypeError, ValueError):
                return (str(a) > str(b)) if op == "gt" else (str(a) < str(b))

        # --- sample_random ---
        if op_name == "sample_random":
            if len(params) < 1:
                return False
            node = _pick_conn(exclude_entities)
            if node is None:
                return False
            role_to_node[params[0]] = node
            exclude_entities.add(node)
            return True

        # --- sample_relation ---
        if op_name == "sample_relation":
            if len(params) < 2 or edge_ids.shape[1] == 0:
                return False
            r_tag, e_role = params[0], params[1]
            r_pid = edge_tag_to_pid.get(r_tag)
            if r_pid is None:
                return False
            r_idx = self._rel_pid_to_idx(r_pid)
            if r_idx is None:
                return False
            candidates = []
            for ei in range(edge_ids.shape[1]):
                if int(rel_types[ei]) == r_idx:
                    h = int(edge_ids[0, ei])
                    if h not in exclude_entities:
                        candidates.append(h)
            if not candidates:
                return False
            node = int(self.rng.choice(candidates))
            role_to_node[e_role] = node
            exclude_entities.add(node)
            return True

        # --- sample_entity_list ---
        if op_name == "sample_entity_list":
            if len(params) < 1 or entity_list is None or len(entity_list) == 0:
                return False
            entity_list_int = [int(x) for x in entity_list]
            candidates = [e for e in entity_list_int if e not in exclude_entities]
            if not candidates:
                return False
            node = int(self.rng.choice(candidates))
            role_to_node[params[0]] = node
            exclude_entities.add(node)
            return True

        # --- or ---
        if op_name == "or":
            if len(params) < 3:
                return False
            r1, r2, out_role = params[0], params[1], params[2]
            n1 = role_to_node.get(r1)
            n2 = role_to_node.get(r2)
            if n1 is None or n2 is None:
                return False
            n1 = int(n1) if not isinstance(n1, list) else int(n1[0])
            n2 = int(n2) if not isinstance(n2, list) else int(n2[0])
            role_to_node[out_role] = [n1, n2]
            return True

        # --- list_forward_find ---
        if op_name == "list_forward_find":
            if len(params) < 3:
                return False
            s_role, r_tag, o_role = params[0], params[1], params[2]
            s_val = role_to_node.get(s_role)
            if s_val is None:
                return False
            s_nodes = [int(x) for x in (s_val if isinstance(s_val, list) else [s_val])]
            if not s_nodes:
                return False
            nd = self._graph_get(self.graph, "neighbourhood_dict", {})
            rel_types = self._graph_get(self.graph, "relation_types", np.array([], dtype=np.int64))
            r_pid = edge_tag_to_pid.get(r_tag)
            r_idx = self._rel_pid_to_idx(r_pid) if r_pid else None
            all_objects: List[Tuple[int, int]] = []
            for s_node in s_nodes:
                neighs = nd.get(int(s_node), {})
                if r_idx is not None:
                    pairs = []
                    for t_node, edge_idxs in neighs.items():
                        for ei in np.asarray(edge_idxs).flatten().tolist():
                            if int(rel_types[ei]) == r_idx:
                                t = int(t_node)
                                pairs.append((t, int(ei)))
                else:
                    rel_to_objs: Dict[int, List[Tuple[int, int]]] = {}
                    for t_node, edge_idxs in neighs.items():
                        for ei in np.asarray(edge_idxs).flatten().tolist():
                            ri = int(rel_types[ei])
                            t = int(t_node)
                            if t not in exclude_entities:
                                rel_to_objs.setdefault(ri, []).append((t, int(ei)))
                    if not rel_to_objs:
                        return False
                    r_idx = int(self.rng.choice(list(rel_to_objs.keys())))
                    pairs = rel_to_objs[r_idx]
                    edge_tag_to_pid[r_tag] = self._rel_idx_to_pid(r_idx)
                    edge_tag_to_label[r_tag] = self._rel_idx_to_label(r_idx)
                for t_node, ei in pairs:
                    _add_er_triple(int(s_node), t_node, r_idx, r_tag)
                    all_objects.append((t_node, ei))
            if not all_objects:
                return False
            seen_t = set()
            obj_list: List[int] = []
            for t_node, _ in all_objects:
                if t_node not in seen_t:
                    seen_t.add(t_node)
                    obj_list.append(t_node)
            role_to_node[o_role] = obj_list
            return True

        # --- forward_find ---
        if op_name == "forward_find":
            s_role, r_tag, o_role = params[0], params[1], params[2]
            s_val = role_to_node.get(s_role)
            if s_val is None or isinstance(s_val, list):
                return False
            s_node = int(s_val)
            neighs = nd.get(s_node, {})
            r_pid = edge_tag_to_pid.get(r_tag)
            r_idx = self._rel_pid_to_idx(r_pid) if r_pid else None
            if r_idx is not None:
                pairs = []
                for t_node, edge_idxs in neighs.items():
                    for ei in np.asarray(edge_idxs).flatten().tolist():
                        if int(rel_types[ei]) == r_idx:
                            t = int(t_node)
                            if t not in exclude_entities:
                                pairs.append((t, int(ei)))
                if not pairs:
                    return False
            else:
                rel_to_objs: Dict[int, List[Tuple[int, int]]] = {}
                for t_node, edge_idxs in neighs.items():
                    for ei in np.asarray(edge_idxs).flatten().tolist():
                        ri = int(rel_types[ei])
                        t = int(t_node)
                        if t not in exclude_entities:
                            rel_to_objs.setdefault(ri, []).append((t, int(ei)))
                if not rel_to_objs:
                    return False
                r_idx = int(self.rng.choice(list(rel_to_objs.keys())))
                pairs = rel_to_objs[r_idx]
                edge_tag_to_pid[r_tag] = self._rel_idx_to_pid(r_idx)
                edge_tag_to_label[r_tag] = self._rel_idx_to_label(r_idx)
            chosen, chosen_ei = pairs[0]
            _add_er_triple(s_node, chosen, r_idx, r_tag)
            role_to_node[o_role] = chosen
            exclude_entities.add(chosen)
            return True

        # --- forward_find_attr ---
        if op_name == "forward_find_attr":
            if len(params) < 3:
                return False
            s_role, attr_tag, val_role = params[0], params[1], params[2]
            s_val = role_to_node.get(s_role)
            if s_val is None or isinstance(s_val, list):
                return False
            s_node = int(s_val)
            attrs = attr_by_node.get(s_node, [])
            if not attrs:
                return False
            attr_idx, attr_label, _ = self.rng.choice(attrs)
            attrs_same = [(i, a, v) for i, a, v in attrs if str(a) == str(attr_label)]
            edge_tag_to_pid[attr_tag] = str(attr_label)
            edge_tag_to_label[attr_tag] = str(attr_label)
            first_val = None
            for _i, _a, val in attrs_same:
                _add_attr_triple(s_node, str(attr_label), val, attr_tag)
                if first_val is None:
                    first_val = str(val)
            role_to_value[val_role] = first_val or ""
            return True

        # --- reverse_find ---
        if op_name == "reverse_find":
            if len(params) < 3:
                return False
            s_role, r_tag, o_role = params[0], params[1], params[2]
            o_val = role_to_node.get(o_role)
            if o_val is None or isinstance(o_val, list):
                return False
            o_node = int(o_val)
            r_pid = edge_tag_to_pid.get(r_tag)
            r_idx = self._rel_pid_to_idx(r_pid) if r_pid else None
            if r_idx is not None:
                pairs = []
                for ei in range(edge_ids.shape[1]):
                    if int(edge_ids[1, ei]) == o_node and int(rel_types[ei]) == r_idx:
                        h = int(edge_ids[0, ei])
                        if h not in exclude_entities:
                            pairs.append((h, ei))
                if not pairs:
                    return False
            else:
                rel_to_subjs: Dict[int, List[Tuple[int, int]]] = {}
                for ei in range(edge_ids.shape[1]):
                    if int(edge_ids[1, ei]) == o_node:
                        ri = int(rel_types[ei])
                        h = int(edge_ids[0, ei])
                        if h not in exclude_entities:
                            rel_to_subjs.setdefault(ri, []).append((h, ei))
                if not rel_to_subjs:
                    return False
                r_idx = int(self.rng.choice(list(rel_to_subjs.keys())))
                pairs = rel_to_subjs[r_idx]
                edge_tag_to_pid[r_tag] = self._rel_idx_to_pid(r_idx)
                edge_tag_to_label[r_tag] = self._rel_idx_to_label(r_idx)
            chosen, chosen_ei = pairs[0]
            _add_er_triple(chosen, o_node, r_idx, r_tag)
            role_to_node[s_role] = chosen
            exclude_entities.add(chosen)
            return True

        # --- sample_by_relation ---
        if op_name == "sample_by_relation":
            if len(params) < 3:
                return False
            s_role, r_tag, o_role = params[0], params[1], params[2]
            r_pid = edge_tag_to_pid.get(r_tag)
            if r_pid is None:
                return False
            r_idx = self._rel_pid_to_idx(r_pid)
            if r_idx is None:
                return False
            h_to_pairs: Dict[int, List[Tuple[int, int]]] = {}
            for ei in range(edge_ids.shape[1]):
                if int(rel_types[ei]) == r_idx:
                    h, t = int(edge_ids[0, ei]), int(edge_ids[1, ei])
                    if t not in exclude_entities:
                        h_to_pairs.setdefault(h, []).append((t, ei))
            if not h_to_pairs:
                return False
            h_node = int(self.rng.choice(list(h_to_pairs.keys())))
            pairs = h_to_pairs[h_node]
            role_to_node[s_role] = h_node
            exclude_entities.add(h_node)
            for t_node, ei in pairs:
                _add_er_triple(h_node, t_node, r_idx, r_tag)
                exclude_entities.add(t_node)
            role_to_node[o_role] = pairs[0][0]
            return True

        # --- find_relations_between ---
        if op_name == "find_relations_between":
            if len(params) < 3:
                return False
            if len(params) == 3:
                e1_role, r_tag, e2_role = params[0], params[1], params[2]
                n1, n2 = role_to_node.get(e1_role), role_to_node.get(e2_role)
            else:
                e1_role, e2_role = params[0], params[1]
                n1, n2 = role_to_node.get(e1_role), role_to_node.get(e2_role)
            if n1 is None or n2 is None or isinstance(n1, list) or isinstance(n2, list):
                return False
            n1, n2 = int(n1), int(n2)
            r_idxs = []
            neighs = nd.get(n1, {})
            for t_node, edge_idxs in neighs.items():
                if int(t_node) == n2:
                    for ei in np.asarray(edge_idxs).flatten().tolist():
                        r_idxs.append(int(rel_types[ei]))
            seen_r: Set[int] = set()
            for ri in r_idxs:
                if ri not in seen_r:
                    seen_r.add(ri)
            if not seen_r:
                return False
            r_list = sorted(seen_r)
            if len(params) == 3:
                r_tag = params[2]
                r_idx = int(self.rng.choice(r_list))
                edge_tag_to_pid[r_tag] = self._rel_idx_to_pid(r_idx)
                edge_tag_to_label[r_tag] = self._rel_idx_to_label(r_idx)
                path_triples.append({
                    "head": self._node_to_qid(n1),
                    "relation": self._rel_idx_to_pid(r_idx),
                    "tail": self._node_to_qid(n2),
                    "triple_type": "entity_relation",
                    "triple_str": f"<subj> {self._node_to_label(n1)} <obj> {self._node_to_label(n2)} <rel> {self._rel_idx_to_label(r_idx)}",
                    "triple_str_readable": f"{self._node_to_label(n1)} --{self._rel_idx_to_label(r_idx)}--> {self._node_to_label(n2)}",
                })
            else:
                exclude_pid = edge_tag_to_pid.get(params[-1]) if len(params) >= 4 else None
                r_tags = params[2:-1] if len(params) >= 4 else params[2:]
                exclude_idx = self._rel_pid_to_idx(exclude_pid) if exclude_pid else None
                candidates = [ri for ri in r_list if ri != exclude_idx] if exclude_idx is not None else r_list
                if len(candidates) < len(r_tags):
                    return False
                chosen = list(self.rng.sample(candidates, len(r_tags)))
                for r_tag, r_idx in zip(r_tags, chosen):
                    edge_tag_to_pid[r_tag] = self._rel_idx_to_pid(r_idx)
                    edge_tag_to_label[r_tag] = self._rel_idx_to_label(r_idx)
                    path_triples.append({
                        "head": self._node_to_qid(int(n1)),
                        "relation": self._rel_idx_to_pid(r_idx),
                        "tail": self._node_to_qid(int(n2)),
                        "triple_type": "entity_relation",
                        "triple_str": f"<subj> {self._node_to_label(int(n1))} <obj> {self._node_to_label(int(n2))} <rel> {self._rel_idx_to_label(r_idx)}",
                        "triple_str_readable": f"{self._node_to_label(int(n1))} --{self._rel_idx_to_label(r_idx)}--> {self._node_to_label(int(n2))}",
                    })
            return True

        # --- sample_random_value ---
        if op_name == "sample_random_value":
            if len(params) < 1:
                return False
            val_role = params[0]
            v2pe = self._build_value_to_property_entities()
            if not v2pe:
                return False
            val_str = str(self.rng.choice(list(v2pe.keys())))
            role_to_value[val_role] = val_str
            return True

        # --- reverse_find_attr ---
        if op_name == "reverse_find_attr":
            if len(params) < 3:
                return False
            e_role, prop_role, val_role = params[0], params[1], params[2]
            val = role_to_value.get(val_role)
            if val is None:
                return False
            v2pe = self._build_value_to_property_entities()
            candidates = v2pe.get(str(val), [])
            candidates = [(p, e) for p, e in candidates if e not in exclude_entities]
            if not candidates:
                return False
            prop_label, s_node = self.rng.choice(candidates)
            role_to_node[e_role] = s_node
            edge_tag_to_pid[prop_role] = str(prop_label)
            edge_tag_to_label[prop_role] = str(prop_label)
            exclude_entities.add(s_node)
            _add_attr_triple(s_node, str(prop_label), val, prop_role)
            return True

        # --- sample_val_gt / sample_val_lt ---
        if op_name in ("sample_val_gt", "sample_val_lt"):
            if len(params) < 3:
                return False
            ent_role, prop_role, val_role = params[0], params[1], params[2]
            threshold = role_to_value.get(val_role)
            if threshold is None:
                return False
            v2pe = self._build_value_to_property_entities()
            prop_entity_pairs = v2pe.get(str(threshold), [])
            if not prop_entity_pairs:
                return False
            prop_label, _ = self.rng.choice(prop_entity_pairs)
            pvs = self._build_property_values_sorted()
            sorted_list = pvs.get(str(prop_label), [])
            cmp_op = "gt" if op_name == "sample_val_gt" else "lt"
            candidates = [
                eid for v, eid in sorted_list
                if v is not None and eid not in exclude_entities and _safe_cmp(v, threshold, cmp_op)
            ]
            if not candidates:
                return False
            node = int(self.rng.choice(candidates))
            role_to_node[ent_role] = node
            edge_tag_to_pid[prop_role] = str(prop_label)
            edge_tag_to_label[prop_role] = str(prop_label)
            exclude_entities.add(node)
            attr_val = next((v for v, e in sorted_list if e == node), None)
            _add_attr_triple(node, str(prop_label), attr_val, prop_role)
            return True

        self.logger.warning(f"[PathSampler] unknown op: {op_name}")
        return False

    def _sample_one(
        self,
        ops: List[List[Any]],
        sparql_content: str,
        entity_key_roles: Optional[List[str]] = None,
        entity_list: Optional[List[int]] = None,
        max_attempts: int = 100,
    ) -> Optional[Dict[str, Any]]:
        """按依赖顺序执行 ops，最多 max_attempts 次。成功返回 path_info（含 sparql_concrete、path_triples 等），失败返回 None。"""
        if not ops:
            return None
        order = self._compute_op_order(ops)
        ordered_ops = [ops[i] for i in order]

        for _ in range(max_attempts):
            role_to_node: Dict[str, Union[int, List[int]]] = {}
            role_to_value: Dict[str, str] = {}
            edge_tag_to_pid: Dict[str, str] = {}
            edge_tag_to_label: Dict[str, str] = {}
            path_triples: List[Dict[str, Any]] = []
            exclude_entities: Set[int] = set()

            ok = True
            for op in ordered_ops:
                if not self._execute_op(
                    op,
                    role_to_node,
                    role_to_value,
                    edge_tag_to_pid,
                    edge_tag_to_label,
                    path_triples,
                    exclude_entities,
                    entity_list=entity_list,
                ):
                    ok = False
                    break
            if not ok or not path_triples:
                continue

            role_to_label: Dict[str, str] = {}
            for r, n in role_to_node.items():
                if isinstance(n, list):
                    role_to_label[r] = ",".join(self._node_to_label(x) for x in n)
                else:
                    role_to_label[r] = self._node_to_label(n)
            role_to_label.update(role_to_value)
            sparql_concrete = self._fill_sparql(
                sparql_content, role_to_node, role_to_value,
                edge_tag_to_pid, for_readable=False
            )
            sparql_readable = self._fill_sparql(
                sparql_content, role_to_node, role_to_value,
                edge_tag_to_label, for_readable=True, role_to_label=role_to_label
            )
            sparql_candidate = self._fill_sparql_candidate(
                sparql_content,
                entity_key_roles=entity_key_roles or [],
                role_to_value=role_to_value,
                edge_tag_to_label=edge_tag_to_label,
                role_to_label=role_to_label,
            )
            er_triples = [t for t in path_triples if t.get("triple_type") == "entity_relation"]
            attr_triples = [t for t in path_triples if t.get("triple_type") == "attribute"]
            path_nodes_qid = []
            path_relations_pid = []
            seen_nodes: Set[str] = set()
            seen_relations: Set[str] = set()
            path_nodes_label_list = []
            path_relations_label_list = []
            for t in er_triples:
                h, r, tail = t.get("head"), t.get("relation"), t.get("tail")
                if h and r and tail:
                    if h not in seen_nodes:
                        path_nodes_qid.append(h)
                        seen_nodes.add(h)
                    if r not in seen_relations:
                        path_relations_pid.append(r)
                        seen_relations.add(r)
                    if tail not in seen_nodes:
                        path_nodes_qid.append(tail)
                        seen_nodes.add(tail)
            for t in attr_triples:
                h, r, tail = t.get("head"), t.get("relation"), t.get("tail")
                if h and r is not None and tail is not None:
                    literal_id = f"VL:{tail}"
                    if h not in seen_nodes:
                        path_nodes_qid.append(h)
                        seen_nodes.add(h)
                    if r not in seen_relations:
                        path_relations_pid.append(r)
                        seen_relations.add(r)
                    if literal_id not in seen_nodes:
                        path_nodes_qid.append(literal_id)
                        seen_nodes.add(literal_id)
            qid_to_label: Dict[str, str] = {}
            for r, n in role_to_node.items():
                nodes = n if isinstance(n, list) else [n]
                for node in nodes:
                    qid = self._node_to_qid(node)
                    if isinstance(n, list):
                        qid_to_label[qid] = self._node_to_label(node)
                    else:
                        qid_to_label[qid] = role_to_label.get(r, self._node_to_label(node))
            for t in attr_triples:
                tail = t.get("tail")
                if tail is not None:
                    qid_to_label[f"VL:{tail}"] = str(tail)
            path_nodes_label_list = [qid_to_label.get(q, q) for q in path_nodes_qid]
            for r in path_relations_pid:
                lbl = next(
                    (edge_tag_to_label.get(tag, r) for tag, pid in edge_tag_to_pid.items() if pid == r),
                    r,
                )
                path_relations_label_list.append(lbl)

            # 移除 path_triples 中的 triple_str_readable，按 (head, relation, tail) 去重
            seen_triple_key: Set[Tuple[str, str, str]] = set()
            triples_clean = []
            for t in path_triples:
                key = (str(t.get("head", "")), str(t.get("relation", "")), str(t.get("tail", "")))
                if key in seen_triple_key:
                    continue
                seen_triple_key.add(key)
                tc = {k: v for k, v in t.items() if k != "triple_str_readable"}
                triples_clean.append(tc)

            answer_list = self._extract_answer(
                sparql_content, role_to_node, role_to_value, role_to_label
            )

            return {
                "sparql_candidate": sparql_candidate,
                "path_triples": triples_clean,
                "entity_keywords": self._build_entity_keywords(role_to_label, entity_key_roles or []),
                "path_nodes_qid": path_nodes_qid,
                "path_relations_pid": path_relations_pid,
                "path_nodes_label": path_nodes_label_list,
                "path_relations_label": path_relations_label_list,
                "sparql_completed": sparql_candidate,
                "answer": answer_list,
            }
        return None

    def _sample_candidates(
        self,
        ops: List[List[Any]],
        sparql_content: str,
        entity_key_roles: Optional[List[str]] = None,
        entity_list: Optional[List[int]] = None,
        num_candidates: int = 1,
        max_attempts: int = 100,
    ) -> List[Dict[str, Any]]:
        """多次采样得到 num_candidates 个 path_info，按 path_triples 去重，供 LLM 选择。"""
        candidates: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()
        if num_candidates <= 0:
            return candidates
        for _ in range(num_candidates):
            cand = self._sample_one(
                ops, sparql_content,
                entity_key_roles=entity_key_roles,
                entity_list=entity_list,
                max_attempts=max_attempts,
            )
            if not cand:
                continue
            key = tuple(t.get("triple_str", "") for t in cand.get("path_triples", []))
            if key and key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
        return candidates

    def _llm_select_sparql(
        self,
        sparql_candidates: List[str],
    ) -> Optional[Dict[str, Any]]:
        """用 LLM 从多个 SPARQL 候选中选一个，解析 JSON 返回 selected_candidate 下标及 reason。"""
        if not self.llm_serving or not sparql_candidates:
            return None
        num_candidates = len(sparql_candidates)
        prompt = self.prompt_template.build_prompt(
            sparql_list=sparql_candidates,
            num_candidates=num_candidates,
        )
        try:
            llm_out = (
                self.llm_serving.generate_from_input(
                    user_inputs=[prompt],
                    system_prompt=self.prompt_template.build_system_prompt(),
                )
                or [""]
            )[0]
        except Exception as e:
            self.logger.error(f"[PathSampler] LLM select failed: {e}")
            return None

        selected_candidate = None
        reason = None
        try:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_out, re.DOTALL)
            j = json.loads(m.group(1)) if m else json.loads(llm_out.strip())
            selected_candidate = j.get("selected_candidate")
            reason = j.get("reason")
        except Exception:
            selected_candidate = None

        try:
            selected_candidate = int(selected_candidate)
        except Exception:
            selected_candidate = None

        if not selected_candidate or not (1 <= selected_candidate <= num_candidates):
            return None

        return {
            "llm_output": llm_out,
            "num_candidates": num_candidates,
            "selected_candidate": selected_candidate,
            "reason": reason,
        }


    def _node_to_qid(self, node_idx: int) -> str:
        """节点索引转为 QID 字符串（如 Q0001），用于 SPARQL 占位符替换。"""
        if self.node_qids is not None:
            return str(self.node_qids[node_idx])
        return f"Q{node_idx:04d}"

    def _node_to_label(self, node_idx: int) -> str:
        """节点索引转为语义标签（如实体名），用于可读 SPARQL 展示。"""
        lbl = self._graph_get(self.graph, "node_labels", None)
        if lbl is not None and 0 <= node_idx < len(lbl):
            s = str(lbl[node_idx])
            return re.sub(r"\s*\([QP]\d+\)\s*$", "", s).strip() or s
        return self._node_to_qid(node_idx)

    def _rel_idx_to_pid(self, r_idx: int) -> str:
        """关系索引转为 pid 字符串（如 P01），用于 SPARQL 占位符。"""
        if self.relation_pids is not None:
            return str(self.relation_pids[r_idx])
        return f"P{r_idx:04d}"

    def _rel_idx_to_label(self, r_idx: int) -> str:
        """关系索引转为语义标签（如关系名），用于可读 SPARQL 展示。"""
        lbl = self._graph_get(self.graph, "relation_labels", None)
        if lbl is not None and 0 <= r_idx < len(lbl):
            s = str(lbl[r_idx])
            return re.sub(r"\s*\([QP]\d+\)\s*$", "", s).strip() or s
        return self._rel_idx_to_pid(r_idx)

    def _fill_sparql(
        self,
        template: str,
        role_to_node: Dict[str, Union[int, List[int]]],
        role_to_value: Dict[str, str],
        edge_tag_to_val: Dict[str, str],
        for_readable: bool = False,
        role_to_label: Optional[Dict[str, str]] = None,
    ) -> str:
        """将 sparql_content 中的 <role>/<tag> 替换为 QID 或语义名。list 用 <item1,item2,...> 格式。"""
        if not template:
            return template
        out = template
        for role, node_idx in role_to_node.items():
            if for_readable and role_to_label and role in role_to_label:
                val = f"«{role_to_label[role]}»"
            else:
                if isinstance(node_idx, list):
                    val = "<" + ",".join(f"wd:{self._node_to_qid(x)}" for x in node_idx) + ">"
                else:
                    val = f"wd:{self._node_to_qid(node_idx)}"
            out = re.sub(rf"<{re.escape(role)}>", val, out)
        for role, val in role_to_value.items():
            rep = f"«{val}»" if for_readable else f'"{val}"'
            out = re.sub(rf"<{re.escape(role)}>", rep, out)
        for tag, val in edge_tag_to_val.items():
            if for_readable:
                rep = f"«{val}»"
            else:
                rep = f"wdt:{val}" if re.match(r"^P\d+$", str(val)) else str(val)
            out = re.sub(rf"<{re.escape(tag)}>", rep, out)
        return out

    def _fill_sparql_candidate(
        self,
        template: str,
        entity_key_roles: List[str],
        role_to_value: Dict[str, str],
        edge_tag_to_label: Dict[str, str],
        role_to_label: Dict[str, str],
    ) -> str:
        """生成 LLM 候选 SPARQL：entity_key 实体用语义名，其余实体用 ?变量 保留，供 LLM 选择。"""
        if not template:
            return template
        out = template

        # 关系标签替换为语义名
        for tag, val in edge_tag_to_label.items():
            out = re.sub(rf"<{re.escape(tag)}>", f"<{val}>", out)

        # 非 entity_key 且不在 role_to_value 的实体占位符替换为变量
        for role in role_to_label.keys():
            if role in entity_key_roles or role in role_to_value:
                continue
            out = re.sub(rf"<{re.escape(role)}>", f"?{role}", out)

        # entity_key 实体替换为语义名：仅在 WHERE 行内替换变量，保留 SELECT 变量；跳过已在 role_to_value 中的 role
        for role in entity_key_roles:
            if role not in role_to_label or role in role_to_value:
                continue
            role_label = f"<{role_to_label[role]}>"
            out = re.sub(rf"<{re.escape(role)}>", role_label, out)
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if re.search(r"\bSELECT\b", line, re.IGNORECASE):
                    continue
                lines[i] = re.sub(rf"\?{re.escape(role)}\b", role_label, line)
            out = "\n".join(lines)

        # 属性值（role_to_value）使用 Literal 形式 "value"，而非 <value>
        for role, val in role_to_value.items():
            out = re.sub(rf"<{re.escape(role)}>", f'"{val}"', out)

        return out

    @staticmethod
    def _build_entity_keywords(
        role_to_label: Dict[str, str],
        entity_key_roles: List[str],
    ) -> List[str]:
        """从 role_to_label 中提取 entity_key_roles 对应的语义标签列表，供下游（如问题生成）使用。"""
        seen = set()
        out = []
        for r in entity_key_roles:
            lbl = role_to_label.get(r)
            if lbl:
                for part in lbl.split(","):
                    part = part.strip()
                    if part and part not in seen:
                        out.append(part)
                        seen.add(part)
        return out

    def _extract_answer(
        self,
        sparql_content: str,
        role_to_node: Dict[str, Union[int, List[int]]],
        role_to_value: Dict[str, str],
        role_to_label: Dict[str, str],
    ) -> List[str]:
        """从 SELECT 变量和 role_to_node/role_to_value/role_to_label 推导答案列表。"""
        m = re.search(
            r"\bSELECT\s+(?:DISTINCT\s+)?(.*?)(?:\s+WHERE|\s+FROM|$)",
            sparql_content,
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return []
        select_part = m.group(1).strip()
        vars_found = re.findall(r"\?(\w+)", select_part)
        answer_list: List[str] = []
        for var in vars_found:
            if var in role_to_label:
                for part in str(role_to_label[var]).split(","):
                    part = part.strip()
                    if part and part not in answer_list:
                        answer_list.append(part)
            elif var in role_to_value:
                val = str(role_to_value[var])
                if val and val not in answer_list:
                    answer_list.append(val)
            elif var in role_to_node:
                n = role_to_node[var]
                nodes = n if isinstance(n, list) else [n]
                for node in nodes:
                    lbl = self._node_to_label(int(node))
                    if lbl and lbl not in answer_list:
                        answer_list.append(lbl)
        return answer_list

    def sample_batch(
        self,
        sparql_patterns: List[Dict[str, Any]],
        entity_list: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """批量采样入口：遍历 sparql_patterns，解析 graph_pattern 为 ops，按需单采或多候选+LLM 选择，返回 [{path_info}, ...]。"""
        results = []
        for sp in sparql_patterns:
            if sp is None:
                results.append({"path_info": None})
                continue
            content = sp.get("sparql_content") or sp.get("sparql_pattern") or ""
            entity_key_roles = sp.get("entity_key") if isinstance(sp, dict) else None
            if isinstance(entity_key_roles, str):
                entity_key_roles = [entity_key_roles]
            if not isinstance(entity_key_roles, list):
                entity_key_roles = []
            entity_key_roles = [str(r).strip() for r in entity_key_roles if str(r).strip()]
            ent_list = sp.get("entity_list", entity_list) if isinstance(sp, dict) else entity_list

            gp = sp.get("graph_pattern") if isinstance(sp, dict) else None
            ops = self._parse_ops(gp) if gp else []
            if not ops:
                self.logger.warning(
                    f"[PathSampler] graph_pattern empty or invalid, keys={list(sp.keys()) if isinstance(sp, dict) else []}"
                )
                results.append({"path_info": None})
                continue
            path_info = None
            num_candidates = max(1, int(self.num_candidates or 1))
            if self.use_llm and self.llm_serving and num_candidates > 1:
                candidates = self._sample_candidates(
                    ops, content, entity_key_roles=entity_key_roles,
                    entity_list=ent_list, num_candidates=num_candidates,
                    max_attempts=self.max_attempts,
                )
                if candidates:
                    sparql_candidates = [
                        c.get("sparql_candidate") or c.get("sparql_structure_readable") or ""
                        for c in candidates
                    ]
                    selection = self._llm_select_sparql(sparql_candidates)
                    if selection:
                        selected_idx = selection["selected_candidate"] - 1
                        path_info = candidates[selected_idx]
                        path_info["sparql_selection"] = selection
                        path_info["sparql_candidates"] = sparql_candidates
                        path_info["sparql_completed"] = sparql_candidates[selected_idx]
            else:
                path_info = self._sample_one(
                    ops, content, entity_key_roles=entity_key_roles,
                    entity_list=ent_list, max_attempts=self.max_attempts,
                )
                if path_info is not None:
                    path_info["sparql_candidates"] = [
                        path_info.get("sparql_candidate") or path_info.get("sparql_structure_readable") or ""
                    ]
                    path_info["sparql_completed"] = path_info["sparql_candidates"][0]
            if path_info is None:
                self.logger.warning(f"[PathSampler] sampling failed, content_len={len(content)}")
            results.append({"path_info": path_info})
        return results
