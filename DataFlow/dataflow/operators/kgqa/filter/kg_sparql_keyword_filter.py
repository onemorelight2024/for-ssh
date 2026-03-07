"""
KGSparqlKeyWordFilter：验证 QA 对中实体关键字是否被改写，从而验证句子的关键语义是否一致。
"""
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGSparqlKeywordFilter(OperatorABC):
    r"""
    验证 QA 对中实体关键字是否被改写，从而验证句子的关键语义是否一致。
    支持从 path_info 自动抽取 entity_keywords。
    所有数据通过 storage 传入：列名固定为 question_rewritten, entity_keywords, path_info；
    输出 synonymity_keep；require_all/verbatim 取自 dataframe.attrs。
    """

    def __init__(self,require_all: bool = True):
        self.logger = get_logger()
        self.processor = KGSparqlKeywordFilterProcessor()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "验证 QA 对中实体关键字是否被改写，从而验证句子的关键语义是否一致。"
        return "Verify entity keywords in QA pairs are not rewritten, ensure key semantics consistent."

    def process_batch(
        self,
        rewritten_questions: List[str],
        entity_keywords_list: List[List[str]],
        require_all: bool = False,
        verbatim: bool = True,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.processor.filter_batch(
            rewritten_questions, entity_keywords_list,
            require_all=require_all, verbatim=verbatim,
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        pass

    def run(self, storage: DataFlowStorage = None):
        r"""所有数据从 storage.read("dataframe") 获取"""
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        require_all = True
        self.processor.require_all = require_all
        self.processor.verbatim = False

        if "question_rewritten" not in dataframe.columns:
            raise ValueError("Missing required column: question_rewritten")

        rewritten_questions = dataframe["question_rewritten"].fillna("").astype(str).tolist()
        path_info_per_row = (
            dataframe.get("path_info", [None] * len(dataframe)).tolist()
            if "path_info" in dataframe.columns
            else [None] * len(dataframe)
        )
        raw_keywords_per_row = (
            dataframe.get("entity_keywords", [None] * len(dataframe)).tolist()
            if "entity_keywords" in dataframe.columns
            else [None] * len(dataframe)
        )

        keywords_per_row = []
        for keywords_for_row, path_info_for_row in zip(raw_keywords_per_row, path_info_per_row):
            if keywords_for_row is not None and (isinstance(keywords_for_row, (list, tuple)) and len(keywords_for_row) > 0):
                keywords_per_row.append(list(keywords_for_row) if isinstance(keywords_for_row, (list, tuple)) else ([keywords_for_row] if keywords_for_row else []))
            elif path_info_for_row is not None and isinstance(path_info_for_row, dict):
                extracted = self.processor.extract_keywords_from_path(path_info_for_row)
                keywords_per_row.append(extracted if extracted else [])
            else:
                keywords_per_row.append([])

        filter_results = self.process_batch(
            rewritten_questions=rewritten_questions,
            entity_keywords_list=keywords_per_row,
            require_all=require_all,
            verbatim=False,
        )
        dataframe["keyword_filter"] = [result_item["keep"] for result_item in filter_results]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return ["keyword_filter"]


class KGSparqlKeywordFilterProcessor:
    r"""比对 entity 是否在重写问题中原样出现，以及所有辅助函数。"""

    def __init__(self):
        self.require_all = False
        self.verbatim = True

    def filter_batch(
        self,
        rewritten_questions: List[str],
        entity_keywords_list: List[List[str]],
        require_all: bool = False,
        verbatim: bool = True,
    ) -> List[Dict[str, Any]]:
        out = []
        for q, kws in zip(rewritten_questions, entity_keywords_list):
            q_raw = (q or "").strip()
            q_compare = q_raw.lower()
            q_compare = self._normalize_text(q_compare)
            kws = [str(w).strip() for w in (kws or []) if w]
            kws = [w.lower() for w in kws]
            kws = [self._normalize_text(w) for w in kws if w]
            if not kws:
                out.append({"keep": True, "matched_keywords": []})
                continue
            matched = [w for w in kws if w in q_compare]
            keep = (len(matched) == len(kws)) if require_all else (len(matched) > 0)
            out.append({"keep": keep, "matched_keywords": matched})
        return out

    @staticmethod
    def extract_keywords_from_path(path_info: Dict) -> List[str]:
        """优先使用 entity_keywords（SPARQL 中出现的自然语言名），其次 path_nodes_label/path_nodes_qid。"""
        entity_keywords = path_info.get("entity_keywords")
        if entity_keywords and isinstance(entity_keywords, (list, tuple)):
            return [str(x).strip() for x in entity_keywords if x]
        labels = list(path_info.get("path_nodes_label") or path_info.get("path_nodes_qid") or [])
        return [str(x).strip() for x in labels if x]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """去掉空格与标点，便于关键词匹配。"""
        return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
