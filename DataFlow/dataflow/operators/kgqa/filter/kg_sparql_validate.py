"""
review完啦
KGSparqlValidate：将 SPARQL 在图上执行，验证返回答案是否与预期一致。
"""
import re
import pandas as pd
from typing import Dict, List, Optional, Any
from rdflib import Graph, Namespace, URIRef, Literal
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGSparqlValidate(OperatorABC):
    r"""
    将 SPARQL 在图上执行，验证返回答案是否与预期一致。
    所有数据通过 storage 传入；
    列名固定为 sparql_completed, answer, path_info；
    输出列为 is_valid, exec_answers, failure_reason。
    """

    def __init__(self):
        self.logger = get_logger()
        self.processor = KGSparqlValidateProcessor()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "将 SPARQL 在图上执行，验证返回答案是否与预期一致。"
        return "Execute SPARQL on graph, validate returned answers match expected."

    def process_batch(
        self,
        sparql_completed_list: list,
        answers: list,
        path_infos: list,
    ) -> List[Dict[str, Any]]:
        raw_data = []
        for sq, ans, pi in zip(sparql_completed_list, answers, path_infos):
            raw_data.append({
                "sparql_query": sq,
                "answer": ans,
                "path_info": pi,
            })
        return self.processor.validate_queries(raw_data)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        pass

    def run(self, storage: DataFlowStorage = None):
        dataframe = storage.read("dataframe")
        if "path_info" not in dataframe.columns:
            raise ValueError("Missing required column: path_info")

        path_infos = dataframe["path_info"].tolist()
        sparql_completed_list = []
        answers = []
        for idx, pi in enumerate(path_infos):
            sq = dataframe.at[idx, "sparql_completed"] if "sparql_completed" in dataframe.columns else None
            if sq is None and isinstance(pi, dict):
                sq = pi.get("sparql_completed")
            sparql_completed_list.append(sq)
            ans = dataframe.at[idx, "answer"] if "answer" in dataframe.columns else None
            if ans is None and isinstance(pi, dict):
                ans = pi.get("answer")
            answers.append(ans)

        outputs = self.process_batch(
            sparql_completed_list, answers, path_infos
        )

        dataframe["is_valid"] = [o.get("is_valid") for o in outputs]
        dataframe["exec_answers"] = [o.get("exec_answers") for o in outputs]
        dataframe["failure_reason"] = [o.get("failure_reason") for o in outputs]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return ["is_valid", "exec_answers", "failure_reason"]


class KGSparqlValidateProcessor:
    r"""SPARQL 执行验证逻辑及所有辅助函数。"""

    def validate_queries(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for data in raw_data:
            sparql_query = data.get("sparql_query")
            expected_answer = data.get("answer")
            path_info = data.get("path_info")
            if not path_info:
                results.append({
                    "is_valid": False,
                    "exec_answers": None,
                    "failure_reason": "缺少path_info，无法构建子图",
                })
                continue
            sparql_query = self._clean_sparql_query(sparql_query)
            sparql_query = self._replace_labels_in_sparql(sparql_query, path_info)
            sparql_query = self._normalize_sparql_for_local_graph(sparql_query)
            graph = self._build_subgraph_from_path_info(path_info)
            if sparql_query is None or len(sparql_query.strip()) == 0:
                results.append({
                    "is_valid": False,
                    "exec_answers": None,
                    "failure_reason": "SPARQL查询为空或清洗失败",
                })
                continue
            try:
                def normalize_answer(a):
                    a = str(a).strip()
                    a = re.sub(r"^(wd|wdt):", "", a)
                    return a.upper()

                qres = graph.query(sparql_query)
                exec_answers = []
                for row in qres:
                    if hasattr(row, "answer"):
                        row_answer = str(row.answer).split("/")[-1].strip()
                        if row_answer:
                            exec_answers.append(row_answer)
                    else:
                        for var in qres.vars:
                            row_answer = str(row[var]).split("/")[-1].strip()
                            if row_answer:
                                exec_answers.append(row_answer)
                            break
                if len(exec_answers) == 0:
                    results.append({
                        "is_valid": False,
                        "exec_answers": None,
                        "failure_reason": "在图上执行SPARQL查询无结果",
                    })
                    continue
                is_valid = True
                failure_reason = None
                if expected_answer is not None and str(expected_answer).strip():
                    if isinstance(expected_answer, list):
                        expected_list = expected_answer
                    else:
                        expected_list = [expected_answer]
                    path_info = path_info or {}
                    path_nodes_qid = path_info.get("path_nodes_qid") or []
                    path_nodes_label = path_info.get("path_nodes_label") or []
                    label_to_qid = {
                        str(lbl): str(qid)
                        for lbl, qid in zip(path_nodes_label, path_nodes_qid)
                        if lbl is not None and qid is not None
                    }
                    expected_list = [label_to_qid.get(str(a), a) for a in expected_list]
                    llm_norm = [normalize_answer(a) for a in expected_list]
                    exec_norm = [normalize_answer(a) for a in exec_answers]
                    if len(set(llm_norm) & set(exec_norm)) == 0:
                        is_valid = False
                        failure_reason = f"答案不匹配（answer: {expected_answer}，执行结果: {exec_answers[:5]}）"
                results.append({
                    "is_valid": is_valid,
                    "exec_answers": exec_answers,
                    "failure_reason": failure_reason,
                })
            except Exception as e:
                results.append({
                    "is_valid": False,
                    "exec_answers": None,
                    "failure_reason": f"执行异常: {e}",
                })
        return results

    @staticmethod
    def _replace_labels_in_sparql(
        sparql: Optional[str],
        path_info: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if sparql is None or not sparql.strip() or not path_info:
            return sparql
        s = str(sparql)
        path_nodes_qid = path_info.get("path_nodes_qid") or []
        path_nodes_label = path_info.get("path_nodes_label") or []
        path_relations_pid = path_info.get("path_relations_pid") or []
        path_relations_label = path_info.get("path_relations_label") or []
        label_to_qid = {
            str(lbl): str(qid)
            for lbl, qid in zip(path_nodes_label, path_nodes_qid)
            if lbl is not None and qid is not None
        }
        rel_label_to_pid = {
            str(lbl): str(pid)
            for lbl, pid in zip(path_relations_label, path_relations_pid)
            if lbl is not None and pid is not None
        }
        for label, qid in label_to_qid.items():
            if str(qid).startswith("VL:"):
                s = re.sub(r"<\s*" + re.escape(label) + r"\s*>", f'"{label}"', s)
            else:
                s = re.sub(r"<\s*" + re.escape(label) + r"\s*>", f"wd:{qid}", s)
        for label, pid in rel_label_to_pid.items():
            s = re.sub(r"<\s*" + re.escape(label) + r"\s*>", f"wdt:{pid}", s)
        return s

    @staticmethod
    def _build_subgraph_from_path_info(
        path_info: Optional[Dict[str, Any]],
    ) -> Graph:
        graph = Graph()
        if not path_info:
            return graph
        entity_uri = "http://example.org/entities/"
        relation_uri = "http://example.org/relations/"
        wd = Namespace(entity_uri)
        wdt = Namespace(relation_uri)
        triples = path_info.get("path_triples") or []
        path_nodes_qid = path_info.get("path_nodes_qid") or []
        path_nodes_label = path_info.get("path_nodes_label") or []
        path_relations_pid = path_info.get("path_relations_pid") or []
        path_relations_label = path_info.get("path_relations_label") or []
        label_to_qid = {
            str(lbl): str(qid)
            for lbl, qid in zip(path_nodes_label, path_nodes_qid)
            if lbl is not None and qid is not None
        }
        rel_label_to_pid = {
            str(lbl): str(pid)
            for lbl, pid in zip(path_relations_label, path_relations_pid)
            if lbl is not None and pid is not None
        }

        def _safe_id(text: str) -> str:
            s = (text or "").strip()
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^\w\-\.]+", "_", s)
            return s or "UNKNOWN"

        def _entity_uri(val: str) -> URIRef:
            key = label_to_qid.get(val, val)
            return URIRef(wd[str(key if key else _safe_id(val))])

        def _relation_uri(val: str) -> URIRef:
            key = rel_label_to_pid.get(val, val)
            return URIRef(wdt[str(key if key else _safe_id(val))])

        def _parse_triple_str(s: str):
            s = (s or "").strip().rstrip(".")
            m = re.search(r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)$", s)
            if m:
                return ("entity_relation", m.group(1).strip(), m.group(3).strip(), m.group(2).strip())
            m = re.search(r"<entity>\s*(.*?)\s*<attribute>\s*(.*?)\s*<value>\s*(.*)$", s)
            if m:
                return ("attribute", m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            return None

        for t in triples:
            if not isinstance(t, dict):
                continue
            h = t.get("head")
            r = t.get("relation")
            tail = t.get("tail")
            triple_type = t.get("triple_type")
            if h and r and tail is not None:
                if triple_type == "attribute":
                    graph.add((URIRef(wd[str(h)]), URIRef(wdt[str(r)]), Literal(tail)))
                else:
                    graph.add((URIRef(wd[str(h)]), URIRef(wdt[str(r)]), URIRef(wd[str(tail)])))
                continue
            triple_str = t.get("triple_str")
            parsed = _parse_triple_str(triple_str) if triple_str else None
            if not parsed:
                continue
            p_type, subj, rel, obj = parsed
            if p_type == "attribute":
                graph.add((_entity_uri(subj), _relation_uri(rel), Literal(obj)))
            else:
                graph.add((_entity_uri(subj), _relation_uri(rel), _entity_uri(obj)))
        return graph

    @staticmethod
    def _clean_sparql_query(sparql: Optional[str]) -> Optional[str]:
        if sparql is None:
            return None
        s = str(sparql).strip()
        if not s:
            return None
        m = re.search(r"```(?:\w*)\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
        return s if s else None

    @staticmethod
    def _normalize_sparql_for_local_graph(
        sparql: Optional[str],
    ) -> Optional[str]:
        if sparql is None or not sparql.strip():
            return sparql
        s = str(sparql).strip()
        entity_uri = "http://example.org/entities/"
        relation_uri = "http://example.org/relations/"
        entity_base = entity_uri.rstrip("/")
        relation_base = relation_uri.rstrip("/")
        s = re.sub(
            r"<" + re.escape(entity_base) + r"/?([QP]\d+)>",
            r"wd:\1", s, flags=re.IGNORECASE
        )
        s = re.sub(
            r"<" + re.escape(relation_base) + r"/?([QP]\d+)>",
            r"wdt:\1", s, flags=re.IGNORECASE
        )
        if ("wd:" in s or "wdt:" in s) and "PREFIX" not in s.upper():
            prefix_block = f"PREFIX wd: <{entity_uri}>\nPREFIX wdt: <{relation_uri}>\n"
            s = prefix_block + s
        return s