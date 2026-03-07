"""
KGQuestionRewriter：输入 question，改写为 rewritten_question，并输出改写策略。
"""
import json
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.kgqa import QuestionRewriterPrompt


@OPERATOR_REGISTRY.register()
class KGQuestionRewriter(OperatorABC):
    r"""
    重写器：输入 question，改写为 rewritten_question。当前 LLM-driven。
    """

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "重写器：输入 QA 对，改写 question；当前 LLM-driven。"
        return "Rewriter: input QA pair, rewrite question; currently LLM-driven."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        pass

    def run(self, storage: DataFlowStorage = None):
        r"""数据从 storage.read("dataframe") 读取。"""
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        qs = dataframe["question"].fillna("").astype(str).tolist()
        qa_pairs = [{"question": q} for q in qs]

        processor = KGQuestionRewriterProcessor(
            llm_serving=self.llm_serving,
            prompt_template=QuestionRewriterPrompt(),
        )
        outputs = processor.rewrite_batch(qa_pairs)
        dataframe["rewritten_question"] = [
            o.get("question_rewritten") or o.get("question") for o in outputs
        ]
        dataframe["rewrite_strategies"] = [o.get("strategies") for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return ["rewritten_question", "rewrite_strategies"]


class KGQuestionRewriterProcessor:
    r"""重写逻辑：rule 或 LLM；LLM 时用 prompt_template.build_prompt。"""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        prompt_template: QuestionRewriterPrompt = QuestionRewriterPrompt(),
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
    def rewrite_batch(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._prompt_rewrite(p.get("question", "")) for p in qa_pairs]


    def _prompt_rewrite(self, q: str, answer: Any = None) -> Dict[str, Any]:
        prompt = self.prompt_template.build_prompt(
            question=str(q).strip(),
            answer=answer,
        )
        out = (
            self.llm_serving.generate_from_input(
                user_inputs=[prompt],
                system_prompt=self.prompt_template.build_system_prompt(),
            )
            or [""]
        )[0]
        rewritten, strategies = self._parse_rewritten_question(out)
        return {
            "question": q,
            "question_rewritten": rewritten,
            "strategies": strategies,
        }

    @staticmethod
    def _parse_rewritten_question(llm_out: str) -> tuple:
        text = (llm_out or "").strip()
        if not text:
            return "", None
        try:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            payload = m.group(1) if m else text
            j = json.loads(payload)
            if isinstance(j, dict) and j.get("question"):
                question = str(j.get("question")).strip()
                strategies = j.get("strategies")
                return question, strategies
        except Exception:
            pass
        first_line = text.split("\n")[0].strip()
        return first_line, None
