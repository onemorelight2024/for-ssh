## Wikontic Status

Wikontic outputs are not currently organized as final JSON result files under the repo.
The main outputs are being written into MongoDB databases.

Known status at packaging time:

- `gpt-4o-mini`: previously completed coverage was effectively `1999/2000` visible sample IDs, with one sample yielding empty extraction.
- `gpt-3.5-turbo`: still running, so not packaged here as a final file result.
- `deepseek-v3`: no file-exported final bundle was found here during packaging.

Relevant configs:

- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_inference.yaml`
- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_missing7_inference.yaml`
- `/home/liuxuem/dataflow-benchmark/Wikontic/inference_and_eval/configs/l2_musique_missing3_inference.yaml`
