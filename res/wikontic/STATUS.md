## Wikontic Status

Wikontic outputs were exported from MongoDB databases into JSON result bundles.

Raw exported files were moved to `../_intermediate/wikontic/`:

- `l2_wikontic_gpt4omini_onto.json`
  - Source database: `l2_triplets_gpt-4o-mini_onto`
  - Unique sample IDs seen: 1999
- `l2_wikontic_gpt35_onto.json`
  - Source database: `l2_triplets_gpt-3_5-turbo_onto`
  - Unique sample IDs seen: 1994
- `l2_wikontic_deepseek_v3_onto.json`
  - Source database: `l2_triplets_deepseek-v3_onto`
  - Unique sample IDs seen: 1986
- `wikontic_export_summary.json`
  - Summary of exported database names, output files, document counts, and sample coverage.

Eval-ready converted files:

- `l2_wikontic_gpt4omini_onto_with_kg.json`
  - 2000 source records
  - Records with at least one final Wikontic relation: 1997
- `l2_wikontic_gpt35_onto_with_kg.json`
  - 2000 source records
  - Records with at least one final Wikontic relation: 1989
- `l2_wikontic_deepseek_v3_onto_with_kg.json`
  - 2000 source records
  - Records with at least one final Wikontic relation: 1980

Each exported JSON contains these Mongo collections:

- `triplets`
- `initial_triplets`
- `ontology_filtered_triplets`
- `filtered_triplets`
- `entity_aliases`

Relevant configs:

- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_inference.yaml`
- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_missing7_inference.yaml`
- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_missing3_inference.yaml`
