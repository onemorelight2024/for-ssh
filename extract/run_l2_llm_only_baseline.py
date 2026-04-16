import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm


if "DF_API_KEY" not in os.environ and os.environ.get("OPENAI_API_KEY"):
    os.environ["DF_API_KEY"] = os.environ["OPENAI_API_KEY"]


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
DATAFLOW_PATH = PROJECT_ROOT / "DataFlow-KG"
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "L2_test_file"
    / "L2_relation_only_with_difficulty_core_plus_modifier_v3_with_kg.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "L2_LLM_only_baseline"
    / "res"
    / "l2_llm_only_baseline_step2.json"
)
if DATAFLOW_PATH.exists() and str(DATAFLOW_PATH) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_PATH))


def load_api_llm_serving_request():
    module_path = DATAFLOW_PATH / "dataflow" / "serving" / "api_llm_serving_request.py"
    spec = importlib.util.spec_from_file_location(
        "dataflow.serving.api_llm_serving_request_direct",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.APILLMServing_request


APILLMServing_request = load_api_llm_serving_request()


ENTITY_SYSTEM_PROMPT = """You are an expert in constructing knowledge graphs. Your task is to extract knowledge entities from the given text.

█ Guidelines:
- You MUST extract ALL entities in the text. Do NOT return only the most important or a subset of entities.
- Perform coreference resolution and canonicalize names where applicable.
- Remove duplicates.
- DO NOT output vague / non-specific references such as:
  "these methods", "this idea", "such systems", "our approach", "it", "they", etc.
- Only output entities that can be clearly and uniquely identified
  (e.g., specific organizations, people, events, concrete concepts, methods, models, software names).

█ Output Format:
Output ONLY a JSON array of strings:
["entity1","entity2",...]"""


ENTITY_USER_PROMPT_TEMPLATE = """Extract all entities from the text.

█ Text
```
{text}
```"""


RELATION_SYSTEM_PROMPT = """You are a knowledge graph construction expert.

Your task:
Generate NEW knowledge graph triples based on:
1) a given list of entity names
2) source texts
3) existing knowledge graph triples
4) common sense or external world knowledge

===============================
ENTITY NORMALIZATION RULE (HARD)
===============================

- Entity names are SYMBOLS, not natural language phrases
- You MUST use entity names EXACTLY as they appear in the entity list
- STRICT STRING MATCH is required

Forbidden modifications include (but are not limited to):
- Adding articles (e.g., "the", "a", "an")
- Changing capitalization
- Adding prefixes or suffixes
- Pluralization or singularization
- Rewriting or paraphrasing entity names

===============================
HARD CONSTRAINTS (MUST FOLLOW)
===============================

1. Existing triple constraint (PAIR-LEVEL NOVELTY):
- You are given a list of EXISTING triples
- You MUST NOT generate any new triple whose
  (subject, object) pair already appears in the existing triples
- This restriction applies EVEN IF the relation is different
- If unsure whether a pair already exists, DO NOT generate it

2. Quality constraint:
- Precision is more important than recall
- If a relation is uncertain, do NOT generate it
- Avoid speculative or controversial facts

=========================
OUTPUT FORMAT (STRICT JSON)
=========================
Return ONLY JSON:
{
  "relations": [
    ["subject", "relation", "object"],
    ["subject", "relation", "object"]
  ]
}"""


RELATION_USER_PROMPT_TEMPLATE = """Please generate NEW knowledge graph triples following all constraints above.

Entity Names:
{entity_list}

Existing Triples (DO NOT repeat subject-object pairs):
{existing_triples}

Source Texts:
{source_texts}

Output STRICT JSON only."""


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_answer_text(response: str) -> str:
    if not isinstance(response, str):
        return ""

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        response = answer_match.group(1)

    return response.replace("```json", "").replace("```", "").strip()


def parse_entities(response: str):
    cleaned = extract_answer_text(response)
    if not cleaned:
        return []

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    results = []
    seen = set()
    for entity in parsed:
        if not isinstance(entity, str):
            continue
        entity = entity.strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        results.append(entity)
    return results


def parse_relations(response: str):
    cleaned = extract_answer_text(response)
    if not cleaned:
        return []

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return []

    triples = parsed.get("relations", parsed.get("triple", []))
    if not isinstance(triples, list):
        return []

    results = []
    seen = set()
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        if not all(isinstance(x, str) for x in triple):
            continue

        subject, relation, obj = [x.strip() for x in triple]
        if not subject or not relation or not obj:
            continue

        key = (subject, relation, obj)
        if key in seen:
            continue
        seen.add(key)
        results.append([subject, relation, obj])

    return results


def generate_with_rerun(
    llm_serving,
    user_inputs,
    system_prompt,
    rerun_rounds=2,
    rerun_backoff_seconds=2.0,
    retry_until_success=False,
    max_rerun_rounds=-1,
    stage_name="stage",
):
    """
    批量调用模型，并对失败项（None/空响应）做重跑。
    返回: (responses, failed_indices)
    """
    responses = [None] * len(user_inputs)
    pending_indices = list(range(len(user_inputs)))

    if retry_until_success:
        # round_idx 从 0 开始计数，max_rerun_rounds = -1 表示无限制
        round_idx = 0
        while pending_indices:
            current_inputs = [user_inputs[i] for i in pending_indices]
            try:
                current_outputs = llm_serving.generate_from_input(
                    user_inputs=current_inputs,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                print(
                    f"[{stage_name}] round {round_idx + 1} 调用异常，"
                    f"{len(pending_indices)} 条待重跑: {type(e).__name__}: {e}"
                )
                current_outputs = [None] * len(current_inputs)

            next_pending = []
            for local_pos, output in enumerate(current_outputs):
                global_pos = pending_indices[local_pos]
                if isinstance(output, str) and output.strip():
                    responses[global_pos] = output
                else:
                    next_pending.append(global_pos)

            pending_indices = next_pending
            if not pending_indices:
                break

            round_idx += 1
            if max_rerun_rounds >= 0 and round_idx > max_rerun_rounds:
                break

            sleep_seconds = rerun_backoff_seconds * (2 ** min(round_idx - 1, 8))
            print(
                f"[{stage_name}] round {round_idx} 后仍有 {len(pending_indices)} 条失败，"
                f"{sleep_seconds:.1f}s 后继续重跑"
            )
            time.sleep(sleep_seconds)

        return responses, pending_indices

    for round_idx in range(rerun_rounds + 1):
        if not pending_indices:
            break

        current_inputs = [user_inputs[i] for i in pending_indices]
        try:
            current_outputs = llm_serving.generate_from_input(
                user_inputs=current_inputs,
                system_prompt=system_prompt,
            )
        except Exception as e:
            print(
                f"[{stage_name}] round {round_idx + 1} 调用异常，"
                f"{len(pending_indices)} 条待重跑: {type(e).__name__}: {e}"
            )
            current_outputs = [None] * len(current_inputs)

        next_pending = []
        for local_pos, output in enumerate(current_outputs):
            global_pos = pending_indices[local_pos]
            if isinstance(output, str) and output.strip():
                responses[global_pos] = output
            else:
                next_pending.append(global_pos)

        if next_pending and round_idx < rerun_rounds:
            sleep_seconds = rerun_backoff_seconds * (2 ** round_idx)
            print(
                f"[{stage_name}] round {round_idx + 1} 后仍有 {len(next_pending)} 条失败，"
                f"{sleep_seconds:.1f}s 后重跑"
            )
            time.sleep(sleep_seconds)

        pending_indices = next_pending

    return responses, pending_indices


def get_text(record, text_key=None):
    if text_key:
        value = record.get(text_key, "")
        return value if isinstance(value, str) else ""

    for key in ("raw_chunk", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_existing_triples(record):
    triples = record.get("triple", [])
    if not isinstance(triples, list):
        return []

    results = []
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        if not all(isinstance(x, str) for x in triple):
            continue
        results.append([x.strip() for x in triple])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-v3.2",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://123.129.219.111:3000/v1/chat/completions",
    )

    parser.add_argument("--text-key", type=str, default=None)
    parser.add_argument("--output-key", type=str, default="triple")
    parser.add_argument("--entity-key", type=str, default="entity")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-entity", action="store_true")
    parser.add_argument(
        "--api-max-workers",
        type=int,
        default=10,
        help="API 并发线程数（传给 APILLMServing_request）",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=5,
        help="单次 API 调用失败后的重试次数（传给 APILLMServing_request）",
    )
    parser.add_argument(
        "--api-request-timeout",
        type=float,
        default=120.0,
        help="单次 API 请求超时秒数（传给 APILLMServing_request）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每批处理的样本数（批内并发调用实体/关系抽取）",
    )
    parser.add_argument(
        "--rerun-rounds",
        type=int,
        default=2,
        help="失败项额外重跑轮数（0 表示不重跑）",
    )
    parser.add_argument(
        "--rerun-backoff-seconds",
        type=float,
        default=2.0,
        help="失败重跑的初始退避秒数（指数退避）",
    )
    parser.add_argument(
        "--retry-until-success",
        action="store_true",
        help="失败项持续重跑直到成功（默认关闭）",
    )
    parser.add_argument(
        "--max-rerun-rounds",
        type=int,
        default=-1,
        help="仅在 --retry-until-success 开启时生效；-1 表示无限重跑",
    )
    args = parser.parse_args()

    if args.api_max_workers <= 0:
        raise ValueError("--api-max-workers must be > 0")
    if args.api_max_retries < 0:
        raise ValueError("--api-max-retries must be >= 0")
    if args.api_request_timeout <= 0:
        raise ValueError("--api-request-timeout must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.rerun_rounds < 0:
        raise ValueError("--rerun-rounds must be >= 0")
    if args.rerun_backoff_seconds < 0:
        raise ValueError("--rerun-backoff-seconds must be >= 0")
    if args.max_rerun_rounds < -1:
        raise ValueError("--max-rerun-rounds must be >= -1")

    records = load_json(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.output.exists() and not args.overwrite:
        output_records = load_json(args.output)
        output_records = output_records[: len(records)]
    else:
        output_records = []

    llm_serving = APILLMServing_request(
        api_url=args.api_url,
        model_name=args.model_name,
        max_workers=args.api_max_workers,
        max_retries=args.api_max_retries,
        request_timeout=args.api_request_timeout,
        temperature=0.0,
    )

    start_idx = len(output_records)
    for batch_start in tqdm(
        range(start_idx, len(records), args.batch_size),
        desc="L2 LLM-only baseline",
    ):
        batch_end = min(batch_start + args.batch_size, len(records))
        batch_records = [dict(records[i]) for i in range(batch_start, batch_end)]
        batch_texts = [get_text(record, args.text_key) for record in batch_records]

        non_empty_local_indices = []
        entity_prompts = []
        for local_idx, text in enumerate(batch_texts):
            if not text.strip():
                continue
            non_empty_local_indices.append(local_idx)
            entity_prompts.append(ENTITY_USER_PROMPT_TEMPLATE.format(text=text))

        entities_by_local_idx = [[] for _ in batch_records]
        if entity_prompts:
            entity_responses, entity_failed_prompt_indices = generate_with_rerun(
                llm_serving=llm_serving,
                user_inputs=entity_prompts,
                system_prompt=ENTITY_SYSTEM_PROMPT,
                rerun_rounds=args.rerun_rounds,
                rerun_backoff_seconds=args.rerun_backoff_seconds,
                retry_until_success=args.retry_until_success,
                max_rerun_rounds=args.max_rerun_rounds,
                stage_name="entity",
            )
            if entity_failed_prompt_indices:
                raise RuntimeError(
                    "entity stage 仍有失败条目未恢复。"
                    "可开启 --retry-until-success 或增加重跑参数。"
                )
            for i, local_idx in enumerate(non_empty_local_indices):
                entity_response = entity_responses[i]
                if isinstance(entity_response, str):
                    entities_by_local_idx[local_idx] = parse_entities(entity_response)
                else:
                    entities_by_local_idx[local_idx] = []

        relation_prompts = []
        relation_local_indices = []
        for local_idx in non_empty_local_indices:
            record = batch_records[local_idx]
            text = batch_texts[local_idx]
            entities = entities_by_local_idx[local_idx]
            relation_prompts.append(
                RELATION_USER_PROMPT_TEMPLATE.format(
                    entity_list=json.dumps(entities, ensure_ascii=False),
                    existing_triples=json.dumps(
                        get_existing_triples(record), ensure_ascii=False
                    ),
                    source_texts=text,
                )
            )
            relation_local_indices.append(local_idx)

        relations_by_local_idx = [[] for _ in batch_records]
        if relation_prompts:
            relation_responses, relation_failed_prompt_indices = generate_with_rerun(
                llm_serving=llm_serving,
                user_inputs=relation_prompts,
                system_prompt=RELATION_SYSTEM_PROMPT,
                rerun_rounds=args.rerun_rounds,
                rerun_backoff_seconds=args.rerun_backoff_seconds,
                retry_until_success=args.retry_until_success,
                max_rerun_rounds=args.max_rerun_rounds,
                stage_name="relation",
            )
            if relation_failed_prompt_indices:
                raise RuntimeError(
                    "relation stage 仍有失败条目未恢复。"
                    "可开启 --retry-until-success 或增加重跑参数。"
                )
            for i, local_idx in enumerate(relation_local_indices):
                relation_response = relation_responses[i]
                if isinstance(relation_response, str):
                    relations_by_local_idx[local_idx] = parse_relations(
                        relation_response
                    )
                else:
                    relations_by_local_idx[local_idx] = []

        for local_idx, record in enumerate(batch_records):
            global_idx = batch_start + local_idx
            text = batch_texts[local_idx]

            if not text.strip():
                if args.save_entity:
                    record[args.entity_key] = []
                record[args.output_key] = []
            else:
                if args.save_entity:
                    record[args.entity_key] = entities_by_local_idx[local_idx]
                record[args.output_key] = relations_by_local_idx[local_idx]

            output_records.append(record)

            if (global_idx + 1) % max(1, args.save_every) == 0:
                save_json(args.output, output_records)

    save_json(args.output, output_records)


if __name__ == "__main__":
    main()
