"""
KGQA 准确性评估算子：有 ground truth 时计算 BLEU-2 等；无时反向生成 SPARQL 再执行验证。
OK
"""
import re
import json
import pandas as pd
from typing import List, Optional, Dict, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.kgqa import SparqlReverseGeneratorPrompt

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
 
from rdflib import Graph as RDFGraph, Namespace, URIRef, Literal


@OPERATOR_REGISTRY.register()
class KGQAValidityEvaluate(OperatorABC):
    r"""
    QA 对准确性评估：
    - 若有 ground truth：计算 BLEU-2、BLEU-4、METEOR、ROUGE-L
    - 若无 ground truth：反向生成 SPARQL 并在图上执行验证（需 pipeline 在 __init__ 中传入 llm_serving、rdf_graph）
    """

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        lang: str = "zh",
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.processor = KGQAValidityEvaluateProcessor(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "QA 准确性评估：有 ground truth 则算 BLEU-2 等；无则反向生成 SPARQL 并执行验证。"
        return "QA validity evaluation: BLEU-2 etc. if ground truth; else reverse-generate SPARQL and validate."

    def process_batch(
        self,
        processor,
        path_infos: List[Dict],
        questions: List[str],
        answers: List[Any],
        sparql_patterns: List[Optional[Dict]],
        entity_uri: Optional[str] = None,
        relation_uri: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> List[Optional[str]]:
        """llmserving 仅用于无 ground truth 时反向生成 SPARQL。返回 SPARQL 字符串列表。"""
        return processor.reverse_generate_batch(
            path_infos=path_infos,
            questions=questions,
            answers=answers,
            sparql_patterns=sparql_patterns,
            entity_uri=entity_uri,
            relation_uri=relation_uri,
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        pass

    def run(self, storage: DataFlowStorage = None):
        r"""
        仅通过 storage.read("dataframe") 和 storage.write(dataframe) 读写。
        列名固定：question、answer、gold_answer、path_info、sparql_pattern。
        无 ground truth 时需在 __init__ 中传入 llm_serving。
        """
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        columns_added = []

        if "gold_answer" in dataframe.columns:
            gt_mask = dataframe["gold_answer"].notna()
        else:
            gt_mask = pd.Series([False] * len(dataframe), index=dataframe.index)

        # 对有 ground truth 的样本计算 BLEU/ROUGE/METEOR
        if gt_mask.any():
            if "question_rewritten" in dataframe.columns:
                question_key = "question_rewritten"
            elif "rewritten_question" in dataframe.columns:
                question_key = "rewritten_question"
            else:
                question_key = "question"
            hyps = dataframe.loc[gt_mask, question_key].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            ).fillna("").astype(str).tolist()
            refs = dataframe.loc[gt_mask, "gold_answer"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            ).fillna("").astype(str).tolist()
            outputs = self.processor.compute_correctness_batch(hyps, refs)
            for k in ["bleu2", "bleu4", "rouge_l"]:
                col = "kgqa_validity_" + k
                if col not in dataframe.columns:
                    dataframe[col] = [None] * len(dataframe)
                dataframe.loc[gt_mask, col] = [o.get(k) for o in outputs]
                if col not in columns_added:
                    columns_added.append(col)

        # 对无 ground truth 的样本走 SPARQL 反向验证
        no_gt_mask = ~gt_mask
        if no_gt_mask.any() and self.llm_serving:
            if "question_rewritten" in dataframe.columns:
                question_key = "question_rewritten"
            elif "rewritten_question" in dataframe.columns:
                question_key = "rewritten_question"
            else:
                question_key = "question"
            path_infos = dataframe.loc[no_gt_mask, "path_info"].tolist()
            questions = dataframe.loc[no_gt_mask, question_key].tolist()
            answers = dataframe.loc[no_gt_mask, "answer"].tolist()
            sparql_patterns = (
                dataframe.loc[no_gt_mask, "sparql_pattern"].tolist()
                if "sparql_pattern" in dataframe.columns
                else [None] * len(path_infos)
            )
            entity_uri = dataframe.attrs.get("entity_uri")
            relation_uri = dataframe.attrs.get("relation_uri")
            processor = KGQAValidityEvaluateProcessor(llm_serving=self.llm_serving)
            reversed_sparqls = self.process_batch(
                processor=processor,
                path_infos=path_infos,
                questions=questions,
                answers=answers,
                sparql_patterns=sparql_patterns,
                entity_uri=entity_uri,
                relation_uri=relation_uri,
            )
            raw_data = []
            for sq, la, pi, sp in zip(reversed_sparqls, answers, path_infos, sparql_patterns):
                raw_data.append({
                    "sparql_query": sq,
                    "answer": la,
                    "path_info": pi,
                    "sparql_pattern": sp,
                    "entity_uri": entity_uri,
                    "relation_uri": relation_uri,
                })
            outputs = processor.validate_batch(raw_data, rdf_graph=None)
            for key in ["is_valid", "exec_answers", "failure_reason"]:
                col = "kgqa_validity_" + key
                if col not in dataframe.columns:
                    dataframe[col] = [None] * len(dataframe)
            dataframe.loc[no_gt_mask, "kgqa_validity_is_valid"] = [o.get("is_valid") for o in outputs]
            dataframe.loc[no_gt_mask, "kgqa_validity_exec_answers"] = [o.get("exec_answers") for o in outputs]
            dataframe.loc[no_gt_mask, "kgqa_validity_failure_reason"] = [o.get("failure_reason") for o in outputs]
            for col in ["kgqa_validity_is_valid", "kgqa_validity_exec_answers", "kgqa_validity_failure_reason"]:
                if col not in columns_added:
                    columns_added.append(col)

        if columns_added:
            output_file = storage.write(dataframe)
            self.logger.info(f"Results saved to {output_file}")
            return columns_added

        self.logger.warning("无 ground truth 且未传入 llm_serving，跳过有效性评估")
        return []

class KGQAValidityEvaluateProcessor:
    r"""调用 LLM、SPARQL 反向生成与验证，以及所有辅助函数。"""

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC] = None,
        lang: str = "zh",
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.lang = (lang or "zh").lower()
        self._jieba = None
        if self.lang.startswith("zh"):
            try:
                import jieba
                self._jieba = jieba
            except Exception:
                try:
                    import sys
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
                    import jieba
                    self._jieba = jieba
                except Exception:
                    self.logger.warning("jieba not available, fallback to char tokenization.")
        self.prompt_template = SparqlReverseGeneratorPrompt()
        self._smoothing = SmoothingFunction().method1
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    def compute_correctness_batch(
        self, hypotheses: List[str], references: List[str]
    ) -> List[Dict[str, Any]]:
        # 计算 BLEU/ROUGE/METEOR，任何单项失败都会被跳过
        results = []
        for hyp, ref in zip(hypotheses, references):
            hyp = (hyp or "").strip()
            ref = (ref or "").strip()
            hyp_tokens = self._tokenize_text(hyp)
            ref_tokens = self._tokenize_text(ref)
            refs_list = [ref_tokens]
            bleu2 = bleu4 = rouge_l = 0.0
            self.logger.info(
                f"Correctness debug | hyp='{hyp}' | ref='{ref}' | "
                f"hyp_tokens={hyp_tokens} | ref_tokens={ref_tokens}"
            )
            hyp_for_rouge = " ".join(hyp_tokens) if hyp_tokens else hyp
            ref_for_rouge = " ".join(ref_tokens) if ref_tokens else ref
            if hyp_tokens and ref_tokens:
                try:
                    bleu2 = sentence_bleu(
                        refs_list, hyp_tokens, weights=(0.5, 0.5), smoothing_function=self._smoothing
                    )
                except Exception:
                    pass
                try:
                    bleu4 = sentence_bleu(
                        refs_list, hyp_tokens,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=self._smoothing,
                    )
                except Exception:
                    bleu4 = bleu2
            if hyp_tokens and ref_tokens:
                if self.lang.startswith("zh"):
                    rouge_l = self._rouge_l_f1(ref_tokens, hyp_tokens)
                elif self._rouge_scorer and hyp_for_rouge and ref_for_rouge:
                    try:
                        rouge_l = self._rouge_scorer.score(ref_for_rouge, hyp_for_rouge)["rougeL"].fmeasure
                    except Exception:
                        pass
            self.logger.info(
                f"Correctness scores | bleu2={bleu2} | bleu4={bleu4} | "
                f"rouge_l={rouge_l}"
            )
            results.append({"bleu2": bleu2, "bleu4": bleu4, "rouge_l": rouge_l})
        return results

    @staticmethod
    def _rouge_l_f1(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
        if not ref_tokens or not hyp_tokens:
            return 0.0
        n, m = len(ref_tokens), len(hyp_tokens)
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            prev = 0
            for j in range(1, m + 1):
                tmp = dp[j]
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = tmp
        lcs_len = dp[m]
        if lcs_len == 0:
            return 0.0
        precision = lcs_len / m
        recall = lcs_len / n
        return (2 * precision * recall) / (precision + recall)

    def _tokenize_text(self, text: str) -> List[str]:
        """分词：中文优先用 jieba；其他语言按空白切分。"""
        if not text:
            return []
        if self.lang.startswith("zh"):
            if self._jieba is not None:
                return [t for t in self._jieba.cut(text) if t.strip()]
            return list(text)
        if re.search(r"\s", text):
            return text.split()
        return list(text)

    def reverse_generate_batch(
        self,
        path_infos: List[Dict],
        questions: List[str],
        answers: List[Any],
        sparql_patterns: List[Optional[Dict]],
        entity_uri: Optional[str] = None,
        relation_uri: Optional[str] = None,
    ) -> List[Optional[str]]:
        """与 KGSparqlReverseGenerateProcessor 对齐：使用 path_entities/path_relations，_normalize 支持 entity_uri/relation_uri。"""
        results = []
        for path_info, question, answer, sparql_pattern in zip(
            path_infos, questions, answers, sparql_patterns
        ):
            raw = {
                "path_info": path_info,
                "question": question,
                "answer": answer,
                "sparql_pattern": sparql_pattern,
                "entity_uri": entity_uri,
                "relation_uri": relation_uri,
            }
            out = self._reverse_generate_one(raw)
            results.append(out.get("reversed_sparql") if out else None)
        return results

    def _reverse_generate_one(self, data: Dict) -> Optional[Dict[str, Any]]:
        """单条反向生成，与 KGSparqlReverseGenerateProcessor._process_one 对齐。"""
        path_info = data.get("path_info")
        question = data.get("question")
        answer = data.get("answer")
        sparql_pattern_dict = data.get("sparql_pattern")
        entity_uri = data.get("entity_uri")
        relation_uri = data.get("relation_uri")
        if not path_info or not path_info.get("path_nodes_qid") or not path_info.get("path_relations_pid"):
            return {"reversed_sparql": None}
        path_nodes_qid = path_info.get("path_nodes_qid", [])
        path_relations_pid = path_info.get("path_relations_pid", [])
        path_nodes_label = path_info.get("path_nodes_label") or []
        path_relations_label = path_info.get("path_relations_label") or []
        answer_str = ", ".join(answer) if isinstance(answer, list) else str(answer)

        path_entities_list = []
        for i in range(len(path_nodes_qid)):
            lbl = path_nodes_label[i] if i < len(path_nodes_label) else path_nodes_qid[i]
            path_entities_list.append({"name": str(lbl), "qid": path_nodes_qid[i]})
        path_relations_list = []
        for i in range(len(path_relations_pid)):
            lbl = path_relations_label[i] if i < len(path_relations_label) else path_relations_pid[i]
            path_relations_list.append({"name": str(lbl), "pid": path_relations_pid[i]})

        sparql_pattern = sparql_pattern_dict.get("sparql_pattern") if sparql_pattern_dict and isinstance(sparql_pattern_dict, dict) else None

        prompt = self.prompt_template.build_prompt(
            question=question,
            answer_str=answer_str,
            path_entities=path_entities_list,
            path_relations=path_relations_list,
            sparql_pattern=sparql_pattern,
        )

        try:
            llm_outputs = self.llm_serving.generate_from_input(
                user_inputs=[prompt],
                system_prompt=self.prompt_template.build_system_prompt(),
            )
            llm_output = llm_outputs[0] if llm_outputs else ""
            sparql_query = self._extract_sparql_from_json(llm_output)
            sparql_query = self._clean_sparql_query(sparql_query)
            sparql_query = self._normalize_sparql_for_local_graph(
                sparql_query, entity_uri=entity_uri, relation_uri=relation_uri
            )
            if not sparql_query or not sparql_query.strip():
                sparql_query = self._generate_sparql_from_path(path_info)
            return {"reversed_sparql": sparql_query}
        except Exception as e:
            self.logger.error(f"Error generating SPARQL: {e}")
            return {"reversed_sparql": self._generate_sparql_from_path(path_info)}

    def validate_batch(
        self,
        raw_data: List[Dict[str, Any]],
        rdf_graph: Any,
    ) -> List[Dict[str, Any]]:
        """与 KGSparqlValidateProcessor.validate_queries 对齐。"""
        results = []
        for data in raw_data:
            sparql_query = data.get("sparql_query")
            expected_answer = data.get("answer")
            entity_uri = data.get("entity_uri")
            relation_uri = data.get("relation_uri")
            sparql_query = self._clean_sparql_query(sparql_query)
            sparql_query = self._replace_labels_in_sparql(sparql_query, data.get("path_info"))
            sparql_query = self._normalize_sparql_for_local_graph(
                sparql_query, entity_uri=entity_uri, relation_uri=relation_uri
            )
            # 若没有传入全图，则基于 path_info 构建子图
            graph = rdf_graph
            if graph is None:
                graph = self._build_subgraph_from_path_info(
                    data.get("path_info"),
                    entity_uri=entity_uri,
                    relation_uri=relation_uri,
                )
            try:
                graph_size = len(graph)
            except Exception:
                graph_size = None
            self.logger.info(
                f"SPARQL query (normalized): {sparql_query if sparql_query else 'EMPTY'}"
            )
            self.logger.info(f"Subgraph size: {graph_size}")
            if graph_size and graph_size > 0:
                try:
                    sample_triples = []
                    for i, t in enumerate(graph.triples((None, None, None))):
                        sample_triples.append(t)
                        if i >= 2:
                            break
                    if sample_triples:
                        self.logger.info(f"Subgraph sample triples: {sample_triples}")
                except Exception:
                    pass
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
                        "failure_reason": "在子图上执行SPARQL查询无结果",
                    })
                    continue
                is_valid = True
                failure_reason = None
                if expected_answer is not None and str(expected_answer).strip():
                    if isinstance(expected_answer, list):
                        expected_list = expected_answer
                    else:
                        expected_list = [expected_answer]
                    path_info = data.get("path_info") or {}
                    path_nodes_qid = path_info.get("path_nodes_qid") or []
                    path_nodes_label = path_info.get("path_nodes_label") or []
                    label_to_qid = {
                        str(lbl): str(qid)
                        for lbl, qid in zip(path_nodes_label, path_nodes_qid)
                        if lbl is not None and qid is not None
                    }
                    expected_list = [
                        label_to_qid.get(str(a), a) for a in expected_list
                    ]
                    llm_norm = [normalize_answer(a) for a in expected_list]
                    exec_norm = [normalize_answer(a) for a in exec_answers]
                    if len(set(llm_norm) & set(exec_norm)) == 0:
                        is_valid = False
                        failure_reason = (
                            f"答案不匹配（answer: {expected_answer}，执行结果: {exec_answers[:5]}）"
                        )
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
        """将 SPARQL 中的 <label> 替换为 wd:Qxx / wdt:Pxx（基于 path_info 的映射）。"""
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
            s = re.sub(r"<\s*" + re.escape(label) + r"\s*>", f"wd:{qid}", s)
        for label, pid in rel_label_to_pid.items():
            s = re.sub(r"<\s*" + re.escape(label) + r"\s*>", f"wdt:{pid}", s)
        return s

    @staticmethod
    def _build_subgraph_from_path_info(
        path_info: Optional[Dict[str, Any]],
        entity_uri: Optional[str] = None,
        relation_uri: Optional[str] = None,
    ):
        """基于 path_info 的三元组构建临时 RDF 子图。"""
        graph = RDFGraph()
        if not path_info:
            return graph
        entity_uri = entity_uri or "http://example.org/entities/"
        relation_uri = relation_uri or "http://example.org/relations/"
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
        """与 KGSparqlValidateProcessor 对齐。"""
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
    def _extract_sparql_from_json(text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        try:
            start = text.find("{")
            if start < 0:
                return None
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end <= start:
                return None
            obj = json.loads(text[start:end])
            if isinstance(obj, dict):
                return obj.get("sparql") or obj.get("query") or obj.get("sparql_query")
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def _normalize_sparql_for_local_graph(
        sparql: Optional[str],
        entity_uri: Optional[str] = None,
        relation_uri: Optional[str] = None,
    ) -> Optional[str]:
        """与 KGSparqlValidateProcessor / KGSparqlReverseGenerateProcessor 对齐。"""
        if sparql is None or not sparql.strip():
            return sparql
        s = str(sparql).strip()
        entity_uri = entity_uri or "http://example.org/entities/"
        relation_uri = relation_uri or "http://example.org/relations/"
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

    @staticmethod
    def _generate_sparql_from_path(path_info: Dict) -> str:
        path_nodes_qid = path_info.get("path_nodes_qid", [])
        path_relations_pid = path_info.get("path_relations_pid", [])
        num_hops = path_info.get("num_hops", 0)
        where_triples = []
        start_entity = path_nodes_qid[0] if path_nodes_qid else "?entity1"
        for i in range(num_hops):
            subject = f"wd:{start_entity}" if i == 0 else f"?entity{i}"
            predicate = f"wdt:{path_relations_pid[i]}" if i < len(path_relations_pid) else "wdt:?relation1"
            obj = "?answer" if i == num_hops - 1 else f"?entity{i + 1}"
            where_triples.append(f"  {subject} {predicate} {obj} .")
        where_clause = "\n".join(where_triples)
        return f"""PREFIX wd: <http://example.org/entities/>
PREFIX wdt: <http://example.org/relations/>
SELECT ?answer
WHERE {{
{where_clause}
}}"""
