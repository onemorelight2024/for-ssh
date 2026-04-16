import argparse
import json
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_l2_llm_only_baseline import (
    DEFAULT_INPUT_PATH,
    ENTITY_SYSTEM_PROMPT,
    ENTITY_USER_PROMPT_TEMPLATE,
    RELATION_SYSTEM_PROMPT,
    RELATION_USER_PROMPT_TEMPLATE,
    extract_answer_text,
    get_existing_triples,
    get_text,
    load_json,
    parse_entities,
    parse_relations,
    save_json,
)


DEFAULT_OUTPUT_PATH = (
    CURRENT_DIR / "res" / "l2_vllm_open_source_baseline_step2.json"
)


class VLLMOpenAIClient:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key_env: str = "VLLM_API_KEY",
        temperature: float = 0.0,
        timeout: int = 300,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env, "EMPTY")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_json = response.json()
        return (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--text-key", type=str, default=None)
    parser.add_argument("--output-key", type=str, default="triple")
    parser.add_argument("--entity-key", type=str, default="entity")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-entity", action="store_true")
    parser.add_argument("--api-key-env", type=str, default="VLLM_API_KEY")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    records = load_json(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.output.exists() and not args.overwrite:
        output_records = load_json(args.output)
        output_records = output_records[: len(records)]
    else:
        output_records = []

    client = VLLMOpenAIClient(
        api_url=args.api_url,
        model_name=args.model_name,
        api_key_env=args.api_key_env,
        temperature=0.0,
        timeout=args.timeout,
    )

    start_idx = len(output_records)
    for idx in tqdm(range(start_idx, len(records)), desc="L2 vLLM baseline"):
        record = dict(records[idx])
        text = get_text(record, args.text_key)

        if not text.strip():
            if args.save_entity:
                record[args.entity_key] = []
            record[args.output_key] = []
            output_records.append(record)
            continue

        entity_user_prompt = ENTITY_USER_PROMPT_TEMPLATE.format(text=text)
        entity_response = client.generate(
            system_prompt=ENTITY_SYSTEM_PROMPT,
            user_prompt=entity_user_prompt,
        )
        entities = parse_entities(entity_response)

        relation_user_prompt = RELATION_USER_PROMPT_TEMPLATE.format(
            entity_list=json.dumps(entities, ensure_ascii=False),
            existing_triples=json.dumps(
                get_existing_triples(record), ensure_ascii=False
            ),
            source_texts=text,
        )
        relation_response = client.generate(
            system_prompt=RELATION_SYSTEM_PROMPT,
            user_prompt=relation_user_prompt,
        )

        if args.save_entity:
            record[args.entity_key] = entities
        record[args.output_key] = parse_relations(relation_response)
        output_records.append(record)

        if (idx + 1) % max(1, args.save_every) == 0:
            save_json(args.output, output_records)

    save_json(args.output, output_records)


if __name__ == "__main__":
    main()
