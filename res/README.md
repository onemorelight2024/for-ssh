## Result Bundle

This folder collects the final result files that are ready to upload.

### Included

- `autoschema/`
  - 6 final AutoSchemaKG result files:
  - `l2_autoschemakg_deepseek_stage1_with_kg.json`
  - `l2_autoschemakg_deepseek_stage2_with_kg.json`
  - `l2_autoschemakg_gpt35_stage1_with_kg.json`
  - `l2_autoschemakg_gpt35_stage2_with_kg.json`
  - `l2_autoschemakg_gpt4omini_stage1_with_kg.json`
  - `l2_autoschemakg_gpt4omini_stage2_with_kg.json`
- `all_llm/`
  - `all_llm_v3_with_kg.json`
- `cc/`
  - `cccode_with_kg.json`
- `wikontic/`
  - 3 eval-ready Wikontic files:
  - `l2_wikontic_deepseek_v3_onto_with_kg.json`
  - `l2_wikontic_gpt35_onto_with_kg.json`
  - `l2_wikontic_gpt4omini_onto_with_kg.json`
- `_intermediate/`
  - Raw or intermediate files moved out of the final result set.

### AutoSchemaKG Stages

- `stage1_with_kg`
  - Result saved after `kg.run_extraction()` and `json_saver1(cfg)` in [run.py](/home/liuxuem/dataflow-benchmark/AutoSchemaKG/run.py).
  - This is the pre-concept-merging ablation.
- `stage2_with_kg`
  - Result reconstructed from the stage2 concept merge artifacts produced after `generate_concept_csv_temp()` and `create_concept_csv()`.
  - This is the post-concept-merging ablation.
- Older `*_triple.json` and earlier `*_with_kg.json` files were the wrong packaging for the final bundle and were moved to `_intermediate/autoschema_old_wrong/`.

### Wikontic Status

- Wikontic results were exported from MongoDB into JSON files under `wikontic/`.
- Coverage at export time:
  - `gpt-4o-mini`: 1999 unique sample IDs
  - `gpt-3.5-turbo`: 1994 unique sample IDs
  - `deepseek-v3`: 1986 unique sample IDs

### Notes

- This bundle currently contains the files that are directly ready to upload.
- I did not move or stop any running Wikontic jobs.
- Files ending in `_with_kg.json` are ready for `eval/evaluate_fact_entailment.py`.
- The final result set contains 11 JSON files:
  - 6 AutoSchemaKG files
  - 1 all-LLM file
  - 1 CC file
  - 3 Wikontic files
- All 11 final JSON files currently load successfully and each contains 2000 items.
