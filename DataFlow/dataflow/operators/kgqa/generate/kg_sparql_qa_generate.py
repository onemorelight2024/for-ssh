"""
KGSparqlQAGenerate：根据完整 SPARQL 生成 QA 对中的问题。

必要输入字段：sparql_completed（完整的 SPARQL 查询）。
输出字段：question（生成的问题）。
"""
import json
import re
import pandas as pd
from typing import Dict, List, Optional, Any

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.kgqa import SparqlCompletedQuestionPrompt


@OPERATOR_REGISTRY.register()
class KGSparqlQAGenerate(OperatorABC):
    r"""
    根据完整 SPARQL 生成对应问题。
    输入：dataframe["sparql_completed"]
    输出：dataframe["question"]
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.qa_processor = KGSparqlQAGenerateProcessor(
            llm_serving=self.llm_serving,
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "根据完整 SPARQL 生成对应的问题。"
        return "Generate questions from completed SPARQL queries."

    def process_batch(
        self,
        sparql_completed_list: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        r"""批量：根据完整 SPARQL 生成问题。"""
        qa_proc = self.qa_processor
        qa_outputs = qa_proc.llm_query(
            [{"sparql_completed": s} for s in sparql_completed_list]
        )
        return qa_outputs

    def _validate_dataframe(self, dataframe: pd.DataFrame, sparql_input_key: str):
        if sparql_input_key not in dataframe.columns:
            raise ValueError(f"Missing column: {sparql_input_key}")

    def run(self, storage: DataFlowStorage = None):
        r"""数据从 storage.read("dataframe") 读取。"""
        dataframe = storage.read("dataframe")
        sparql_input_key = "sparql_completed"
        output_question_key = "question"
        self._validate_dataframe(dataframe, sparql_input_key)

        self.qa_processor = KGSparqlQAGenerateProcessor(
            llm_serving=self.llm_serving,
        )

        sparql_completed_list = [
            str(row.get(sparql_input_key)) if row.get(sparql_input_key) is not None else ""
            for _, row in dataframe.iterrows()
        ]
        qa_outputs = self.process_batch(sparql_completed_list)

        if output_question_key not in dataframe.columns:
            dataframe[output_question_key] = None

        for idx, qa in enumerate(qa_outputs):
            if idx >= len(dataframe.index):
                break
            row_idx = dataframe.index[idx]
            if qa:
                dataframe.at[row_idx, output_question_key] = qa.get("question")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [output_question_key]


class KGSparqlQAGenerateProcessor:
    r"""基于完整 SPARQL 用 LLM 生成问题。"""

    def __init__(
        self,
        llm_serving: LLMServingABC,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = SparqlCompletedQuestionPrompt()

    def llm_query(self, raw_data: List[Dict]) -> List[Dict]:
        results = []
        for data in raw_data:
            sparql_completed = (data.get("sparql_completed") or "").strip()
            prompt = self._build_prompt(sparql_completed)
            try:
                llm_out = (
                    self.llm_serving.generate_from_input(
                        user_inputs=[prompt],
                        system_prompt=self.prompt_template.build_system_prompt(),
                    )
                    or [""]
                )[0]
                question = None
                try:
                    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_out, re.DOTALL)
                    j = json.loads(m.group(1)) if m else json.loads(llm_out.strip())
                    question = j.get("question")
                except (json.JSONDecodeError, AttributeError, KeyError):
                    qm = re.search(r"Question\s*:\s*(.+?)(?:\n|$)", llm_out, re.IGNORECASE)
                    question = (
                        (qm.group(1).strip() if qm else llm_out.strip().split("\n")[0].strip())
                        if llm_out.strip()
                        else None
                    )
                results.append({"question": question, "llm_output": llm_out})
            except Exception as e:
                self.logger.error(f"Error generating QA: {e}")
                results.append({"question": None, "llm_output": None})
        return results

    def _build_prompt(self, sparql_completed: str) -> str:
        return self.prompt_template.build_prompt(sparql_completed=sparql_completed)

