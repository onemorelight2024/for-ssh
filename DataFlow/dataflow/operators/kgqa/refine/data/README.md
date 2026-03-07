# Refine 算子测试数据

- `data_*.jsonl`：测试输入数据
- `expected_*.jsonl`：预期输出结构

| 文件 | 算子 | 说明 |
|------|------|------|
| data_rewriter.jsonl | KGQuestionRewriter | question, llm_answer；输出 question_rewritten。llm_serving 通过算子 __init__ 传入 |
