# Filter 算子测试数据

- `data_*.jsonl`：测试输入数据
- `expected_*.jsonl`：预期输出结构（参考用，部分字段如 reversed_sparql 依赖 LLM 输出）

| 文件 | 算子 | 说明 |
|------|------|------|
| data_filter_reverse_generate.jsonl | KGSparqlReverseGenerate | path_info, question, llm_answer, sparql_pattern |
| data_filter_keyword.jsonl | KGSparqlKeyWordFilter | question_rewritten, entity_keywords 或 path_info |
| data_filter_validate.jsonl | KGSparqlValidate | reversed_sparql, llm_answer, path_info, sparql_pattern；需传入 rdf_graph |
