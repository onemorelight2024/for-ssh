"""
KGQA相关的Prompt模板
"""
from dataflow.core.prompt import PromptABC
from dataflow.utils.registry import PROMPT_REGISTRY
import json
from typing import Dict, List, Optional, Any

@PROMPT_REGISTRY.register()
class SparqlCandidateSelectionPrompt(PromptABC):
    """SPARQL候选选择的Prompt"""

    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return "你是一个知识图谱问答数据生成助手，擅长从多个 SPARQL 路径中评估并选择最有语义价值的查询。"

    def build_prompt(self, sparql_list: List[str], num_candidates: int) -> str:
        prompt = f"""我需要从多个 SPARQL 路径中选择一个最有价值的查询来生成QA对。
【输入信息】：
候选 SPARQL 列表（共 {num_candidates} 个）：
{sparql_list}
【选择标准】：
1. **语义合理性**：关系链条应逻辑连贯，实体与关系之间的连接符合常识；避免无意义的跳转、循环或断裂；subject-object-relation 三元组在领域内应有明确语义。
2. **信息价值**：优先选择信息量更高、能覆盖更多实体和关系类型的查询；关系应具有代表性，能体现知识图谱的核心语义；避免过于泛化（如仅查类型）或过于琐碎（如仅查单属性）的查询。
3. **可问性**：基于该 SPARQL 能够生成自然、明确、无歧义的自然语言问题；用户应能直观理解问题意图；问题应具体可答，而非模糊或过于抽象。
4. **多样性**：若多个候选在语义上相近，优先选择能体现不同关系类型或不同推理路径的查询，以丰富后续生成的 QA 多样性。

请从候选 1 到候选 {num_candidates} 中选择一个最有价值的候选，并简要说明选择理由。
【输出格式】
严格按照以下JSON格式输出：
{{
    "selected_candidate": [候选编号，整数，范围1-{num_candidates}],
    "reason": "[选择理由，字符串]"
}}"""
        return prompt




@PROMPT_REGISTRY.register()
class SparqlReverseGeneratorPrompt(PromptABC):
    """SPARQL反向生成的Prompt"""
    
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return "你是一名精通知识图谱与语义解析的 SPARQL 专家，擅长将自然语言问题准确映射为可执行的 SPARQL 1.1 查询语句。"
    
    def build_prompt(
        self,
        question: str,
        answer_str: str,
        path_entities: List[str],
        path_relations: List[str],
        sparql_pattern: Optional[str] = None
    ) -> str:
        """构建SPARQL反向生成prompt
        
        Args:
            question: 自然语言问题
            answer_str: 答案实体字符串
            path_entities: 路径实体名称列表
            path_relations: 路径关系名称列表
            sparql_pattern: SPARQL模式（可选）
        """
        path_entities_str = json.dumps(path_entities, ensure_ascii=False, indent=2)
        path_relations_str = json.dumps(path_relations, ensure_ascii=False, indent=2)
        if sparql_pattern:
            prompt = f"""你的任务是将自然语言问题准确映射为可执行的 SPARQL 1.1 查询语句。

【输入信息】
1. **自然语言问题**：{question}
   - 用户提出的原始问题，需理解其语义意图（如查询某实体的属性、某关系下的对象、多跳推理路径等）。
2. **答案实体**：{answer_str}
   - 问题对应的正确答案实体，SPARQL 查询的结果应能返回该实体；用于校验查询目标是否正确。
3. **SPARQL模式约束**（必须严格遵循此结构，不可改动）：{sparql_pattern}
   - 预定义的查询骨架，内含占位符（如实体名、关系名）；你只需用具体名称替换占位符，不得增删变量、括号、FILTER、OPTIONAL 等结构。
4. **可用的节点字典**：{path_entities_str}
   - 当前路径上所有可用的实体名称列表；只能从该字典中选取实体填入占位符，不得使用未列出的实体。
5. **可用的关系字典**：{path_relations_str}
   - 当前路径上所有可用的关系/属性名称列表；只能从该字典中选取关系填入占位符，不得使用未列出的关系。

【操作步骤】
1. **语义映射**：仔细阅读自然语言问题，识别其询问的主体、关系、目标；在"可用节点字典"和"可用关系字典"中逐一找到与问题语义对应的实体名和关系名；若存在多义或歧义，优先选择与答案实体最相关的组合。
2. **模式填充**：将 SPARQL 模式约束中的每个占位符替换为步骤 1 中选定的具体名称；实体与关系均用 <名称> 形式包裹，例如 <实体A>、<关系R>；保持占位符原有的位置和数量，一一对应替换。
3. **格式校验**：确认输出为合法的 SPARQL 1.1 语法；变量名（如 ?x、?y）保持不变；SELECT、WHERE、括号层级、三元组顺序等与模式约束完全一致；禁止使用 QID/PID、wd:/wdt: 等前缀，仅使用语义名称。

【严格约束】
1. **结构不可变**：绝对忠实于 SPARQL 模式约束，仅进行占位符的精确替换；严禁增删变量、修改括号、调整三元组顺序、添加或删除 FILTER/OPTIONAL/UNION 等子句。
2. **名称来源**：所有实体和关系名称必须来自给定的"可用节点字典"和"可用关系字典"，不得臆造或使用同义词（除非字典中明确列出）。
3. **输出纯净**：只输出一个合法的 JSON 对象，格式为 {{"sparql_query": "..."}}；禁止包含任何解释、思考过程、Markdown 代码块标记（如 ```json）或多余换行。
4. **可执行性**：生成的 SPARQL 应语法正确、变量绑定清晰，能够在本体/图上实际执行并返回答案实体。

请按照以下严格的 JSON 格式输出：
{{"sparql_query": "完整的SPARQL查询语句"}}
"""
        else:
            prompt = f"""将自然语言问题映射为可执行的 SPARQL 1.1 查询语句。

【输入信息】
1. **自然语言问题**：{question}
   - 用户提出的原始问题，需理解其语义意图（如查询某实体的属性、某关系下的对象、多跳推理路径等）。
2. **答案实体**：{answer_str}
   - 问题对应的正确答案实体，SPARQL 查询的结果应能返回该实体；用于校验查询目标是否正确。
3. **可用的节点字典**：{path_entities_str}
   - 当前路径上所有可用的实体名称列表；只能从该字典中选取实体，不得使用未列出的实体。
4. **可用的关系字典**：{path_relations_str}
   - 当前路径上所有可用的关系/属性名称列表；只能从该字典中选取关系，不得使用未列出的关系。

【操作步骤】
1. **语义映射**：仔细阅读自然语言问题，识别其询问的主体、关系、目标；在"可用节点字典"和"可用关系字典"中逐一找到与问题语义对应的实体名和关系名。
2. **构建查询**：根据问题意图构建合法的 SPARQL 1.1 查询；实体与关系均用 <名称> 形式包裹；SELECT 变量需与 WHERE 中的绑定一致。
3. **格式校验**：确认输出为合法的 SPARQL 1.1 语法；禁止使用 QID/PID、wd:/wdt: 等前缀，仅使用语义名称。

【严格约束】
1. **名称来源**：所有实体和关系名称必须来自给定的"可用节点字典"和"可用关系字典"，不得臆造或使用未列出的同义词。
2. **输出纯净**：只输出一个合法的 JSON 对象，格式为 {{"sparql_query": "..."}}；禁止包含任何解释、思考过程、Markdown 代码块标记或多余换行。
3. **可执行性**：生成的 SPARQL 应语法正确、变量绑定清晰，能够在本体/图上实际执行并返回答案实体。

请按照以下严格的 JSON 格式输出：
{{"sparql_query": "完整的SPARQL查询语句"}}
"""
        return prompt

@PROMPT_REGISTRY.register()
class QuestionRewriterPrompt(PromptABC):
    """问题改写 Prompt：输入 QA 对，只改写 question，可带 answer 作上下文防语义偏移。"""

    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return "你是一个语言学专家，擅长为给定问句生成语义完全等价但句法结构不同的改写版本。"

    def build_prompt(self, question: str, answer: Any = None) -> str:
        return f"""
你的任务是根据给定的自然语言问题，生成一个语义完全等价但句法结构不同的改写版本。
【输入】：
原句：{question}
答案：{answer}
【可选择的改写策略】（根据问题内容，选择最适合的策略组合，至少使用1种）：
1. **语态转换（主动/被动）** - 在保持施事-受事关系的前提下，通过改变动词形态和论元位置调整句子视角。
2. **疑问词替换与焦点转移** - 替换wh-词（如why→for what reason）或调整疑问焦点（方式→工具），改变询问视角但不改变询问目标。
3. **分裂句与准分裂句** - 使用"It is/was..."或"What...is/was..."结构将特定成分置于焦点位置进行强调。
4. **话题化与左偏置** - 将非疑问成分移至句首作为话题，调整信息结构以改变句子节奏和关注点。
5. **名词化/动词化** - 在动词短语和名词短语间转换（如"如何分析"→"分析方法是什么"），改变句子的语法中心。
6. **词汇概念分解** - 将复杂动词分解为轻动词+名词化形式（如"solve"→"find solution to"），明确动作的语义结构。
7. **信息结构调整** - 重组已知信息与新信息，通过添加语用框架（"Regarding X,"）或调整成分顺序改变信息流。
8. **语体正式度调整** - 在正式与非正式语体间转换，通过词汇选择、句式复杂度和礼貌标记改变语体特征。
9. **修辞结构变化** - 使用平行结构、反问、省略等修辞手法，增强表达效果但不改变核心命题。
【输出格式】：
只输出 JSON 对象，格式如下：
{{
    "question": "改写后的问题",
    "strategies": ["选择的改写策略","选择的改写策略2","..."]
}}
        """


@PROMPT_REGISTRY.register()
class SparqlCompletedQuestionPrompt(PromptABC):
    """基于完整 SPARQL 生成问题的 Prompt"""

    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        return "你是一个专业的知识图谱问答数据生成器，擅长根据 SPARQL 查询结构生成准确反映语义意图的自然语言问题。"

    def build_prompt(self, sparql_completed: str) -> str:
        return f"""你的任务是根据提供的包含语义信息的 SPARQL 查询结构，生成一个自然、明确、可回答的自然语言问题。

【输入信息】
**包含语义信息的 SPARQL 结构**：{sparql_completed}
   - 该 SPARQL 已包含实体和关系的语义名称；
   - 需从中识别：起始实体（查询的锚点）、关系链（subject→relation→object 的路径）、目标（最终要返回的实体或属性值）；

【生成要求】
1. **语义忠实**：问题必须准确反映 SPARQL 的语义意图，不得遗漏或曲解关系链中的关键信息；目标实体/属性值应与 SELECT 或查询终点一致。
2. **自然可读**：问题应像用户自然提问，句式通顺、无歧义。
3. **隐含路径**：仅使用起始实体和关系描述来隐含表达路径；**不得在问题中写出中间实体的名称**。
4. **目标明确**：问题应明确指出用户在问什么（如某个实体的属性、某关系下的对象、满足某条件的实体列表等）。
5. **长度适中**：问题不宜过长或过短；单句为主，必要时可适当补充限定语以保证清晰。

【严格约束】
1. **输出纯净**：只输出一个合法的 JSON 对象，格式为 {{"question": "..."}}；禁止包含任何解释、思考过程、Markdown 代码块标记（如 ```json）或多余换行。
2. **一问一答**：生成的问题应能通过该 SPARQL 查询得到确定答案；避免模糊、多义或无法从图中回答的问题。

【输出格式】
严格按照以下 JSON 格式输出：
{{
    "question": "生成的自然语言问题"
}}"""
