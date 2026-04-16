"""
事实蕴含评测脚本。

评测流程基于 MINE 的思路：
1. 从已生成的 KG 构建图结构。
2. 对图节点生成 embedding。
3. 用 fact 检索图上下文。
4. 让 LLM 判断检索上下文是否支持该 fact。

在此基础上做了两点增强：
- 保存每个 fact 的评测结果，方便后续难度分析。
- 支持断点续跑，并且服务类错误不会直接记为 0 分。
"""

from dotenv import load_dotenv
import argparse
import dspy
import json
import networkx as nx
import numpy as np
import os
import sys
import time
from typing import Any, Dict, List

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.kg_gen.kg_gen import KGGen



# Reuse the current project configuration
API_KEY = "sk-7r0NxTCXfOIOx3nguqAS5CwoOR9HKcXm5gnjKadL0wssqW2E"
BASE_URL = "http://123.129.219.111:3000/v1"

lm = dspy.LM(
    model="gpt-4o-mini",
    api_key=API_KEY,
    api_base=BASE_URL,
    temperature=0.1,
    max_tokens=4000,
)
dspy.configure(lm=lm)


RETRYABLE_ERROR_MARKERS = (
    "serviceunavailableerror",
    "rate limit",
    "ratelimiterror",
    "apitimeouterror",
    "timeout",
    "apiconnectionerror",
    "connection error",
    "openai_error",
    "503",
    "502",
    "504",
)


class RetryableEvaluationError(Exception):
    """Raised when the evaluation backend is temporarily unavailable."""


class EvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


class ResponseEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateResponse)

    def forward(self, context, correct_answer):
        return self.evaluate(context=context, correct_answer=correct_answer)


evaluator = ResponseEvaluator()


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """Evaluate whether the retrieved context supports the fact."""
    result = evaluator.forward(context=context, correct_answer=correct_answer)
    return int(result.evaluation)


def is_retryable_service_error(error: Exception) -> bool:
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(marker in error_text for marker in RETRYABLE_ERROR_MARKERS)


def evaluate_with_retry(
    correct_answer: str,
    context: str,
    max_retries: int = 3,
    base_delay_seconds: float = 2.0,
) -> int:
    for attempt in range(max_retries + 1):
        try:
            return gpt_evaluate_response(correct_answer, context)
        except Exception as error:
            if not is_retryable_service_error(error):
                raise

            if attempt >= max_retries:
                raise RetryableEvaluationError(str(error)) from error

            delay_seconds = base_delay_seconds * (2 ** attempt)
            print(
                f"    评测服务暂时不可用，准备在 {delay_seconds:.1f}s 后重试 "
                f"({attempt + 1}/{max_retries})"
            )
            time.sleep(delay_seconds)

    raise RetryableEvaluationError("Evaluation backend remained unavailable after retries.")


def iter_fact_objects(item: Dict[str, Any]):
    for quality_level in ["high", "medium", "low", "unknown"]:
        if quality_level not in item.get("relational_facts", {}):
            continue
        for fact_obj in item["relational_facts"][quality_level]:
            yield quality_level, fact_obj


def is_fact_scored(fact_obj: Dict[str, Any]) -> bool:
    return (
        "kg_entailment_score" in fact_obj
        and fact_obj["kg_entailment_score"] is not None
    )


def is_item_fully_scored(item: Dict[str, Any]) -> bool:
    facts_found = False
    for _, fact_obj in iter_fact_objects(item):
        facts_found = True
        if not is_fact_scored(fact_obj):
            return False
    return facts_found


def calculate_item_stats(item: Dict[str, Any]) -> Dict[str, Any]:
    total_facts = 0
    scored_facts = 0
    correct_facts = 0
    service_error_facts = 0
    error_facts = 0

    for _, fact_obj in iter_fact_objects(item):
        total_facts += 1
        status = fact_obj.get("evaluation_status")
        if status == "service_error":
            service_error_facts += 1
        elif status == "error":
            error_facts += 1

        if is_fact_scored(fact_obj):
            scored_facts += 1
            correct_facts += fact_obj["kg_entailment_score"]

    return {
        "total_facts": total_facts,
        "scored_facts": scored_facts,
        "correct_facts": correct_facts,
        "service_error_facts": service_error_facts,
        "error_facts": error_facts,
        "accuracy": correct_facts / scored_facts if scored_facts > 0 else 0,
    }


def calculate_overall_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_facts = 0
    scored_facts = 0
    correct_facts = 0
    service_error_facts = 0
    error_facts = 0

    difficulty_stats = {
        "easy": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
        "medium": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
        "hard": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
    }

    for item in dataset:
        for _, fact_obj in iter_fact_objects(item):
            total_facts += 1
            fact_difficulty = fact_obj.get("difficulty", "unknown")
            status = fact_obj.get("evaluation_status")

            if fact_difficulty in difficulty_stats:
                difficulty_stats[fact_difficulty]["all_facts"] += 1

            if status == "service_error":
                service_error_facts += 1
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["service_error_facts"] += 1
            elif status == "error":
                error_facts += 1
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["error_facts"] += 1

            if is_fact_scored(fact_obj):
                scored_facts += 1
                correct_facts += fact_obj["kg_entailment_score"]
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["total"] += 1
                    difficulty_stats[fact_difficulty]["correct"] += fact_obj[
                        "kg_entailment_score"
                    ]

    for difficulty in difficulty_stats:
        stats = difficulty_stats[difficulty]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "total_facts": total_facts,
        "scored_facts": scored_facts,
        "correct_facts": correct_facts,
        "service_error_facts": service_error_facts,
        "error_facts": error_facts,
        "overall_accuracy": correct_facts / scored_facts if scored_facts > 0 else 0,
        "difficulty_breakdown": difficulty_stats,
    }


def build_result_payload(
    dataset: List[Dict[str, Any]],
    retrieval_k: int,
    retrieval_depth: int,
) -> Dict[str, Any]:
    overall_stats = calculate_overall_stats(dataset)

    simplified_dataset = []
    for item in dataset:
        if "evaluation_stats" not in item:
            continue

        fact_results = []
        for quality_level, fact_obj in iter_fact_objects(item):
            fact_result = {
                "quality_level": quality_level,
                "fact": fact_obj.get("fact"),
                "difficulty": fact_obj.get("difficulty", "unknown"),
                "relational_type": fact_obj.get("relational_type"),
                "confidence": fact_obj.get("confidence"),
                "triples": fact_obj.get("triples", {}),
                "kg_entailment_score": fact_obj.get("kg_entailment_score"),
                "evaluation_status": fact_obj.get("evaluation_status"),
            }

            if fact_obj.get("corrected_fact") is not None:
                fact_result["corrected_fact"] = fact_obj.get("corrected_fact")

            if "retrieval_stats" in fact_obj:
                fact_result["retrieval_stats"] = fact_obj["retrieval_stats"]

            if "error" in fact_obj:
                fact_result["error"] = fact_obj["error"]

            fact_results.append(fact_result)

        simplified_dataset.append(
            {
                "id": item.get("id", "unknown"),
                "title": item.get("title", "unknown"),
                "evaluation_stats": item["evaluation_stats"],
                "fact_results": fact_results,
            }
        )

    return {
        "evaluation_config": {
            "retrieval_model": "all-MiniLM-L6-v2",
            "retrieval_k": retrieval_k,
            "retrieval_depth": retrieval_depth,
            "evaluator_model": "gpt-4o-mini",
        },
        "dataset_stats": simplified_dataset,
        "overall_stats": overall_stats,
    }


def save_result_payload(
    dataset: List[Dict[str, Any]],
    output_path: str,
    retrieval_k: int,
    retrieval_depth: int,
) -> None:
    result = build_result_payload(dataset, retrieval_k, retrieval_depth)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def restore_progress_from_output(dataset: List[Dict[str, Any]], output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0

    with open(output_path, "r", encoding="utf-8") as file:
        previous_result = json.load(file)

    previous_items = {
        item.get("id", "unknown"): item
        for item in previous_result.get("dataset_stats", [])
    }

    restored_items = 0
    for item in dataset:
        item_id = item.get("id", "unknown")
        previous_item = previous_items.get(item_id)
        if previous_item is None:
            continue

        previous_fact_lists = {quality: [] for quality in ["high", "medium", "low", "unknown"]}
        for fact_result in previous_item.get("fact_results", []):
            quality_level = fact_result.get("quality_level", "unknown")
            previous_fact_lists.setdefault(quality_level, []).append(fact_result)

        restored_any = False
        for quality_level in ["high", "medium", "low", "unknown"]:
            current_facts = item.get("relational_facts", {}).get(quality_level, [])
            previous_facts = previous_fact_lists.get(quality_level, [])
            for index, fact_obj in enumerate(current_facts):
                if index >= len(previous_facts):
                    continue

                previous_fact = previous_facts[index]

                if previous_fact.get("error") is not None:
                    fact_obj["error"] = previous_fact["error"]
                    fact_obj["evaluation_status"] = previous_fact.get(
                        "evaluation_status",
                        "service_error",
                    )
                    continue

                if previous_fact.get("kg_entailment_score") is not None:
                    fact_obj["kg_entailment_score"] = previous_fact["kg_entailment_score"]
                    fact_obj["evaluation_status"] = previous_fact.get(
                        "evaluation_status",
                        "scored",
                    )
                    if "retrieval_stats" in previous_fact:
                        fact_obj["retrieval_stats"] = previous_fact["retrieval_stats"]
                    restored_any = True

        item["evaluation_stats"] = calculate_item_stats(item)
        if restored_any:
            restored_items += 1

    return restored_items


def evaluate_single_item(
    item: Dict[str, Any],
    kggen: KGGen,
    retrieval_k: int = 8,
    retrieval_depth: int = 2,
) -> Dict[str, Any]:
    """Evaluate all facts for a single item."""
    print(f"评测条目: {item.get('title', item.get('id', 'Unknown'))}")

    if "extracted_kg" not in item:
        print("  跳过：没有 extracted_kg 字段")
        return item

    try:
        graph = kggen.from_dict(item["extracted_kg"])
        nx_graph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nx_graph)

        print(f"  图结构: {nx_graph.number_of_nodes()} 节点, {nx_graph.number_of_edges()} 边")

        for quality_level in ["high", "medium", "low", "unknown"]:
            if quality_level not in item.get("relational_facts", {}):
                continue

            facts_list = item["relational_facts"][quality_level]
            print(f"  评测 {quality_level} 质量级别的 {len(facts_list)} 个facts")

            for fact_obj in facts_list:
                if is_fact_scored(fact_obj):
                    continue

                fact_text = fact_obj["fact"]

                try:
                    top_nodes, context_edges, context_text = kggen.retrieve(
                        query=fact_text,
                        node_embeddings=node_embeddings,
                        graph=nx_graph,
                        k=retrieval_k,
                        depth=retrieval_depth,
                    )

                    fact_obj["retrieval_stats"] = {
                        "top_k": retrieval_k,
                        "depth": retrieval_depth,
                        "retrieved_node_count": len(top_nodes),
                        "retrieved_context_edge_count": len(context_edges),
                        "retrieved_context_char_length": len(context_text),
                        "retrieved_context_word_count": len(context_text.split()),
                    }

                    evaluation_result = evaluate_with_retry(fact_text, context_text)

                    fact_obj["kg_entailment_score"] = evaluation_result
                    fact_obj["retrieved_context"] = context_text
                    fact_obj["evaluation_status"] = "scored"
                    fact_obj.pop("error", None)

                except RetryableEvaluationError as error:
                    print(f"    评测服务错误，暂不计分: {str(error)}")
                    fact_obj.pop("kg_entailment_score", None)
                    fact_obj["evaluation_status"] = "service_error"
                    fact_obj["error"] = str(error)

                except Exception as error:
                    print(f"    错误：评测fact时出错，暂不计分: {str(error)}")
                    fact_obj.pop("kg_entailment_score", None)
                    fact_obj["evaluation_status"] = "error"
                    fact_obj["error"] = str(error)

        item["evaluation_stats"] = calculate_item_stats(item)
        print(
            "  结果: "
            f"{item['evaluation_stats']['correct_facts']}/"
            f"{item['evaluation_stats']['scored_facts']} = "
            f"{item['evaluation_stats']['accuracy'] * 100:.1f}% "
            f"(服务错误 {item['evaluation_stats']['service_error_facts']}, "
            f"其他错误 {item['evaluation_stats']['error_facts']})"
        )

    except Exception as error:
        print(f"  错误：处理KG时出错: {str(error)}")
        item["evaluation_error"] = str(error)

    return item


def evaluate_dataset(
    dataset_path: str,
    output_path: str,
    retrieval_k: int = 8,
    retrieval_depth: int = 2,
) -> None:
    """Evaluate the whole dataset and checkpoint progress to the output file."""
    print(f"开始评测数据集: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    restored_items = restore_progress_from_output(dataset, output_path)
    if restored_items > 0:
        print(f"从已有结果中恢复了 {restored_items} 个已完成条目的进度")

    kggen = KGGen(retrieval_model="all-MiniLM-L6-v2")
    print(f"数据集包含 {len(dataset)} 个条目")

    for index, item in enumerate(dataset):
        print(f"\n处理第 {index + 1}/{len(dataset)} 个条目")
        if is_item_fully_scored(item):
            item["evaluation_stats"] = calculate_item_stats(item)
            print(f"  跳过：{item.get('title', item.get('id', 'Unknown'))} 已完成")
        else:
            dataset[index] = evaluate_single_item(
                item,
                kggen,
                retrieval_k=retrieval_k,
                retrieval_depth=retrieval_depth,
            )

        save_result_payload(dataset, output_path, retrieval_k, retrieval_depth)

    overall_stats = calculate_overall_stats(dataset)
    save_result_payload(dataset, output_path, retrieval_k, retrieval_depth)

    print(f"\n评测完成，结果已保存到: {output_path}")
    print(
        "整体结果: "
        f"{overall_stats['correct_facts']}/{overall_stats['scored_facts']} = "
        f"{overall_stats['overall_accuracy']:.1%}"
    )
    print(
        "未计分情况: "
        f"服务错误 {overall_stats['service_error_facts']}, "
        f"其他错误 {overall_stats['error_facts']}"
    )


def main():
    parser = argparse.ArgumentParser(description="评测KG生成质量：事实蕴含评测")
    parser.add_argument(
        "--dataset",
        default="dataset/L2_relation_only_with_difficulty_core_plus_modifier_with_kg.json",
        help="数据集文件路径",
    )
    parser.add_argument(
        "--output",
        default="dataset/res/final_L2_evaluation_results.json",
        help="输出结果文件路径",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="检索时选取的 top-k 相关节点数",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="从命中节点向外扩展的图上下文深度",
    )

    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.output,
        retrieval_k=args.k,
        retrieval_depth=args.depth,
    )


if __name__ == "__main__":
    main()
