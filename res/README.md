## Result Bundle

This folder collects the result files that are ready to upload.

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
  - `all_llm_v3.json`
- `cc/`
  - `cccode.json`
- `wikontic/`
  - Status notes only for now.

### Wikontic Status

- Wikontic results are currently stored in MongoDB rather than exported JSON files.
- One `gpt-3.5-turbo` run is still in progress and was intentionally not touched.
- Completed / partial progress should be exported later if you want a file-based bundle for GitHub.

### Notes

- This bundle currently contains the files that are directly ready to upload.
- I did not move or stop any running Wikontic jobs.
