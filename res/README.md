## Result Bundle

This folder collects the final result files that are ready to upload.

### Included

- `autoschema/`
  - 6 final AutoSchemaKG result files:
  - `l2_autoschemakg_deepseek_triple.json`
  - `l2_autoschemakg_deepseek_with_kg.json`
  - `l2_autoschemakg_gpt35_triple.json`
  - `l2_autoschemakg_gpt35_with_kg.json`
  - `l2_autoschemakg_gpt4omini_triple.json`
  - `l2_autoschemakg_gpt4omini_with_kg.json`
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
