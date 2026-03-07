"""
review完了
KGSparqlReverseGenerate：根据 QA 对反向生成 SPARQL，验证语义是否与最初的模板保持一致。
记得把entity补上前缀
"""
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import DIYPromptABC
from dataflow.prompts.kgqa import SparqlReverseGeneratorPrompt


@OPERATOR_REGISTRY.register()
class KGSparqlReverseGenerate(OperatorABC):
    r"""
    根据 QA 对反向生成 SPARQL，验证语义是否与最初的模板保持一致。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = SparqlReverseGeneratorPrompt()
        self.processor = KGSparqlReverseGenerateProcessor(
            llm_serving=llm_serving,
            prompt_template=self.prompt_template,
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "根据 QA 对反向生成 SPARQL，验证语义是否与最初的模板保持一致。"
        return "Reverse generate SPARQL from QA pairs, validate semantics consistent with original template."

    def process_batch(
        self,
        path_infos: list,
        questions: list,
        answers: list,
        sparql_patterns: list,
        sources: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        return self.processor.reverse_generate_batch(path_infos, questions, answers, sparql_patterns)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        pass

    def run(
        self,
        storage: DataFlowStorage = None,
        path_info_key: str = "path_info",
        question_key: str = "rewritten_question",
        answer_key: str = "answer",
        sparql_pattern_key: str = "sparql_pattern",
        output_key: str = "reversed_sparql",
    ):
        r"""注意：数据加载和读写用 storage.read("dataframe") 和 storage.write(dataframe)"""
        self.path_info_key = path_info_key
        self.question_key = question_key
        self.answer_key = answer_key
        self.sparql_pattern_key = sparql_pattern_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        path_infos = dataframe[path_info_key].tolist()
        if question_key not in dataframe.columns and "question_rewritten" in dataframe.columns:
            question_key = "question_rewritten"
        if answer_key not in dataframe.columns and "llm_answer" in dataframe.columns:
            answer_key = "llm_answer"
        questions = dataframe[question_key].tolist()
        answers = dataframe[answer_key].tolist()
        sparql_patterns = (
            dataframe.get(sparql_pattern_key, [None] * len(dataframe)).tolist()
            if sparql_pattern_key in dataframe.columns
            else [None] * len(dataframe)
        )

        outputs = self.process_batch(path_infos, questions, answers, sparql_patterns)
        dataframe[output_key] = [o.get("reversed_sparql") for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]


class KGSparqlReverseGenerateProcessor:
    r"""调用 LLM 反向生成 SPARQL，以及所有辅助函数。"""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        prompt_template: Union[SparqlReverseGeneratorPrompt, DIYPromptABC],
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template

    def reverse_generate_batch(
        self,
        path_infos: list,
        questions: list,
        answers: list,
        sparql_patterns: list,
    ) -> List[Dict[str, Any]]:
        results = []
        for path_info, question, answer, sparql_pattern in zip(
            path_infos, questions, answers, sparql_patterns
        ):
            raw = {
                "path_info": path_info,
                "question": question,
                "answer": answer,
                "sparql_pattern": sparql_pattern,
            }
            out = self._process_one(raw)
            results.append(out)
        return results

    def _process_one(self, data: Dict) -> Dict[str, Any]:
        path_info = data.get("path_info")
        question = data.get("question")
        answer = data.get("answer")
        sparql_pattern_dict = data.get("sparql_pattern")
        path_nodes_qid = path_info.get("path_nodes_qid", [])
        path_relations_pid = path_info.get("path_relations_pid", [])
        path_triples = path_info.get("path_triples", [])
        path_nodes_label = path_info.get("path_nodes_label", [])
        path_relations_label = path_info.get("path_relations_label", [])
        answer_str = ", ".join(answer) if isinstance(answer, list) else str(answer)
        
        path_entities = json.dumps(path_nodes_label or [], ensure_ascii=False, indent=2)
        path_relations = json.dumps(path_relations_label or [], ensure_ascii=False, indent=2)

        sparql_pattern = sparql_pattern_dict.get("sparql_pattern")

        prompt = self.prompt_template.build_prompt(
            question=question,
            answer_str=answer_str,
            path_entities=path_entities,
            path_relations=path_relations,
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
            path_entities_uri = data.get("path_entities_uri") 
            path_relations_uri = data.get("path_relations_uri")
            sparql_query = self._map_labels_to_qp(
                sparql_query,
                path_nodes_label,
                path_nodes_qid,
                path_relations_label,
                path_relations_pid,
            )
            sparql_query = self._normalize_sparql_for_local_graph(
                sparql_query, path_entities_uri, path_relations_uri
            )

            return {"reversed_sparql": sparql_query}
        except Exception as e:
            self.logger.error(f"Error generating SPARQL: {e}")
            return {"reversed_sparql": self._generate_sparql_from_path(path_info)}

    @staticmethod
    def _clean_sparql_query(sparql: Optional[str]) -> Optional[str]:
        s = str(sparql).strip()
        m = re.search(r"```(?:\w*)\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
        return s

    @staticmethod
    def _extract_sparql_from_json(text: str) -> Optional[str]:
        """从 LLM 输出中提取 sparql_query 字段。"""
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return s
        try:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
            payload = m.group(1) if m else s
            j = json.loads(payload)
            if isinstance(j, dict) and j.get("sparql_query"):
                return str(j.get("sparql_query")).strip()
        except Exception:
            pass
        return s

    @staticmethod
    def _normalize_sparql_for_local_graph(
        sparql: Optional[str],
        path_entities_uri: Optional[str] = None,
        path_relations_uri: Optional[str] = None,
    ) -> Optional[str]:
        if sparql is None or not sparql.strip():
            return sparql
        s = str(sparql).strip()
        entity_uri = path_entities_uri or "http://example.org/entities/"
        relation_uri = path_relations_uri or "http://example.org/relations/"
        # 将完整 URI 转为 wd:/wdt: 简写（根据传入的命名空间参数匹配）
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
        # 若使用了 wd:/wdt: 但开头没有命名空间声明，则补上
        if ("wd:" in s or "wdt:" in s) and "PREFIX" not in s.upper():
            prefix_block = f"PREFIX wd: <{entity_uri}>\nPREFIX wdt: <{relation_uri}>\n"
            s = prefix_block + s
        return s

    @staticmethod
    def _map_labels_to_qp(
        sparql: Optional[str],
        node_labels: List[str],
        node_qids: List[str],
        relation_labels: List[str],
        relation_pids: List[str],
    ) -> Optional[str]:
        if sparql is None or not str(sparql).strip():
            return sparql
        ent_map = {str(k).strip(): str(v).strip() for k, v in zip(node_labels or [], node_qids or [])}
        rel_map = {str(k).strip(): str(v).strip() for k, v in zip(relation_labels or [], relation_pids or [])}

        def _repl(match: re.Match) -> str:
            token = match.group(1).strip()
            if token in rel_map:
                return f"wdt:{rel_map[token]}"
            if token in ent_map:
                return f"wd:{ent_map[token]}"
            return match.group(0)

        return re.sub(r"<([^<>]+)>", _repl, str(sparql))

