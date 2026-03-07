"""
KGQA 语义多样性评估算子：基于句法依存树相似度和 self-BLEU 分数的混合打分。
这个文件，后期比较一下sparql相似度和依存树是不是强相关的。
review过
"""
self_bleu_weigh=0.5
dep_sim_weight=0.5

from rapidfuzz import fuzz
import pandas as pd
from typing import List, Optional, Dict, Any
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from fast_bleu import SelfBLEU
import spacy
from rdflib.plugins.sparql import prepareQuery


@OPERATOR_REGISTRY.register()
class KGQASemanticDiversityEvaluate(OperatorABC):
    r"""
    QA 对的语义多样性评估：句法依存树相似度 + self-BLEU 混合打分。
    """

    def __init__(self):
        self.logger = get_logger()
        self.processor = KGQASemanticDiversityEvaluateProcessor()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "QA 对语义多样性评估：句法依存树相似度 + self-BLEU 混合打分。"
        return "QA semantic diversity evaluation: dependency tree similarity + self-BLEU mixed scoring."

    def process_batch(
        self,
        questions: List[str],
        sparql_list: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # 计算整表的 self-BLEU 与依存树相似度（输出单条聚合结果）
        result = self.processor.evaluate_batch(questions, sparql_list)
        return [result]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if "question" not in dataframe.columns and "rewritten_question" not in dataframe.columns:
            raise ValueError("Missing required column: at least one of question or rewritten_question")

    def run(self, storage: DataFlowStorage = None):
        r"""
        仅通过 storage.read("dataframe") 和 storage.write(dataframe) 读写。
        列名固定：question、rewritten_question（可选，优先用于多样性计算）、reversed_sparql（可选）。
        """
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        questions = []
        for idx, row in dataframe.iterrows():
            q = row.get("rewritten_question") or row.get("question") or ""
            questions.append(str(q).strip() if q is not None else "")
        sparql_list = (
            dataframe["reversed_sparql"].fillna("").astype(str).tolist()
            if "reversed_sparql" in dataframe.columns
            else None
        )

        outputs = self.process_batch(questions, sparql_list)
        result = outputs[0] if outputs else {}

        dataframe["kgqa_self_bleu_score"] = result.get("self_bleu_score", 0.0)
        dataframe["kgqa_dependency_similarity"] = result.get("dependency_similarity", 0.0)
        dataframe["kgqa_semantic_diversity_score"] = result.get("combined_diversity_score", 0.0)

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [
            "kgqa_self_bleu_score",
            "kgqa_dependency_similarity",
            "kgqa_semantic_diversity_score",
        ]


class KGQASemanticDiversityEvaluateProcessor:
    r"""self-BLEU、依存树相似度、SPARQL 句法树相似度计算，以及所有辅助函数。"""

    def __init__(
        self
    ):
        self.logger = get_logger()
        self.nlp = spacy.load("en_core_web_sm")

    def evaluate_batch(
        self, questions: List[str], sparql_list: Optional[List[str]] = None
    ) -> Dict[str, float]:
        # 依赖数量：少于 2 条无法计算 self-BLEU/相似度
        self_bleu = self._compute_self_bleu(questions) if len(questions) >= 2 else 0.0
        #dep_sim 是依存树相似度，可选
        dep_sim = self._compute_dep_similarity_spacy(questions) if len(questions) >= 2 else 0.0
        #sp_sims = self._compute_sparql_tree_similarity_batch(sparql_list) if sparql_list and len(sparql_list) >= 2 else [0.0]
        #sp_sim_avg = sum(sp_sims) / len(sp_sims) if sp_sims else 0.0


        combined = self_bleu_weigh * (1.0 - self_bleu) + dep_sim_weight * (1.0 - dep_sim)
        return {
            "self_bleu_score": self_bleu,
            "dependency_similarity": dep_sim,
            "combined_diversity_score": combined,
        }

    def _compute_self_bleu(self, questions: List[str]) -> float:

        tokens = [(q or "").split() for q in questions]
        #这里1-gram和2-gram的权重都是0.5，此处待定，我感觉1-gram会容易导致词频重复，因为question一般比较短
        sb = SelfBLEU(tokens, {"bigram": (0.5, 0.5)})
        scores = sb.get_score()["bigram"]
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_dep_similarity_spacy(self, questions: List[str]) -> float:
        # 依存树相似度（句法结构的近似度）
        if not self.nlp or len(questions) < 2:
            return 0.0
        trees = []
        for q in questions:
            q = (q or "").strip()
            if not q:
                trees.append("")
                continue
            try:
                doc = self.nlp(q)
                # 只用 dep 类型，不受节点内容/下标影响
                parts = [tok.dep_ for tok in doc]
                trees.append(" ".join(sorted(parts)))
            except Exception:
                trees.append("")
        sims = []
        for i in range(len(trees)):
            for j in range(len(trees)):
                if i != j and trees[i] and trees[j]:
                    sims.append(fuzz.ratio(trees[i], trees[j]) / 100.0)
        return sum(sims) / len(sims) if sims else 0.0
'''
    def _compute_sparql_tree_similarity_batch(self, sparql_list: List[str]) -> List[float]:
        # 使用 self-BLEU 风格：每条 SPARQL 以其余为参考，计算结构 n-gram 重叠度
        try:
            from rdflib.plugins.sparql import prepareQuery
        except ImportError:
            return [0.0] * len(sparql_list)

        norm_strings = []
        for s in sparql_list:
            s = (s or "").strip()
            if not s:
                norm_strings.append(None)
                continue
            try:
                norm = self._sparql_to_normalized_tree_str(s)
                norm_strings.append(norm)
            except Exception:
                norm_strings.append(None)

        valid_indices = [i for i, n in enumerate(norm_strings) if n is not None]
        if len(valid_indices) < 2:
            return [0.0] * len(sparql_list)

        # 将归一化树字符串按空白切分为 token 序列，用于 self-BLEU
        tokens = [norm_strings[i].split() for i in valid_indices]
        sb = SelfBLEU(tokens, {"bigram": (0.5, 0.5)})
        scores = sb.get_score()["bigram"]

        out = [0.0] * len(sparql_list)
        for idx, i in enumerate(valid_indices):
            out[i] = scores[idx]
        return out

    def _sparql_to_normalized_tree_str(self, sparql: str) -> str:
        query = prepareQuery(sparql)
        var_map = {}
        return self._algebra_to_normalized_str(query.algebra, var_map)

    def _algebra_to_normalized_str(self, node: Any, var_map: dict) -> str:
        if node is None:
            return "null"
        if hasattr(node, "name"):
            name = getattr(node, "name", "")
            parts = [f"({name}"]
            if hasattr(node, "keys"):
                for k in sorted(node.keys()):
                    if k.startswith("_"):
                        continue
                    v = node[k]
                    parts.append(f" {k}={self._algebra_to_normalized_str(v, var_map)}")
            elif hasattr(node, "__iter__") and not isinstance(node, (str, bytes)):
                try:
                    for i, v in enumerate(node):
                        parts.append(f" [{i}]={self._algebra_to_normalized_str(v, var_map)}")
                except (TypeError, AttributeError):
                    pass
            parts.append(")")
            return "".join(parts)
        if hasattr(node, "n3"):
            n3 = node.n3()
            if "?" in n3:
                if node not in var_map:
                    var_map[node] = f"?v{len(var_map)}"
                return var_map[node]
            if hasattr(node, "toPython") and callable(node.toPython):
                try:
                    return str(node.toPython())
                except Exception:
                    pass
            s = str(node)
            if "/" in s:
                return s.split("/")[-1].split("#")[-1].strip()
            return s
        if isinstance(node, (list, tuple)):
            return "[" + ",".join(self._algebra_to_normalized_str(x, var_map) for x in node) + "]"
        return str(node)
'''