"""
AI 提示词配置
统一管理所有 AI 对话相关的系统提示词，便于维护和修改
"""

# AI Chat 系统提示词
AI_CHAT_SYSTEM_PROMPT = """
你是一个专业的数据分析科研助手。当前用户在 Theta 主题分析平台上操作。

你的能力：
- 数据清洗与预处理
- 主题模型 (ETM) 训练与评估
- 智能对话与 AI 科研助手
- 任务中心 · 异步训练与监控
- 可视化与结果导出

请根据用户的问题和当前上下文，提供专业、准确的回答。如果用户提供的是中文，请用中文回复。
"""

# DashScope 模型配置
DASHSCOPE_MODEL = "qwen-plus"

# 流式输出配置
STREAM_RESPONSE = False
AI_CHAT_SYSTEM_PROMPT_MULTI = """
你是一个专业的数据分析科研助手。当前用户在 Theta 主题分析平台上操作。

你的能力：
- 数据清洗与预处理
- 主题模型 (ETM) 训练与评估
- 智能对话与 AI 科研助手
- 任务中心 · 异步训练与监控
- 可视化与结果导出

当用户提供图片或文档附件时，请仔细分析内容并给出专业的解读和建议。如果用户提供的是中文，请用中文回复。
"""

DASHSCOPE_VL_MODEL = "qwen-vl-plus"


# Chart vision interpretation prompts
# Few-shot examples can be added here later without changing the API route.
CHART_ANALYSIS_SYSTEM_PROMPT = (
    "你是科研图表解读助手，专门解读 THETA 主题模型输出的图表。"
    "只输出最终中文解读，不输出推理过程，不输出 <think> 标签，不输出英文分析。"
    "必须用完整句子收尾，不能以半句话、逗号或未完成的项目符号结束。"
)

CHART_ANALYSIS_PROMPT_TEMPLATE = (
    "请只用{language}解读这张 THETA 主题模型分析图表。\n"
    "图表名称：{chart_name}\n"
    "分析类型：{analysis_type}\n\n"
    "输出要求：\n"
    "1. 先说明图表展示的核心信息；\n"
    "2. 指出最值得关注的趋势、差异或异常；\n"
    "3. 给出对研究者有用的结论；\n"
    "4. 控制在 150 到 220 字之间；\n"
    "5. 避免编造图中不存在的数据；\n"
    "6. 不要输出推理过程，不要输出 <think> 标签；\n"
    "7. 不要使用 Markdown 标题或加粗符号；\n"
    "8. 最后一句必须是完整结论，不能中途截断。"
)

CHART_ANALYSIS_FEW_SHOT_EXAMPLES: list[dict] = [
    # Example format for future expansion:
    # {
    #     "chart_name": "主题占比分布.png",
    #     "chart_description": "饼图显示各主题在语料中的占比。",
    #     "analysis": "该图展示了主题分布并不均衡，少数主题占据主要比例。建议优先解读高占比主题，并检查低占比主题是否为噪声或细分议题。"
    # }
]


def build_chart_analysis_messages(
    *,
    chart_name: str,
    analysis_type: str,
    language: str,
    image_data_url: str,
    image_detail: str = "low",
    few_shot_examples: list[dict] | None = None,
) -> list[dict]:
    """Build OpenAI-compatible multimodal messages for chart interpretation."""
    output_language = "中文" if language.startswith("zh") else "English"
    examples = few_shot_examples if few_shot_examples is not None else CHART_ANALYSIS_FEW_SHOT_EXAMPLES

    messages: list[dict] = [
        {"role": "system", "content": CHART_ANALYSIS_SYSTEM_PROMPT},
    ]

    for example in examples:
        example_chart_name = example.get("chart_name", "示例图表")
        example_description = example.get("chart_description", "")
        example_user_text = (
            f"请解读这张示例 THETA 图表。\n"
            f"图表名称：{example_chart_name}\n"
            f"图表说明：{example_description}"
        )
        example_content: list[dict] = [{"type": "text", "text": example_user_text}]
        if example.get("image_url"):
            example_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": example["image_url"], "detail": image_detail},
                }
            )
        messages.append({"role": "user", "content": example_content})
        messages.append({"role": "assistant", "content": example.get("analysis", "")})

    prompt = CHART_ANALYSIS_PROMPT_TEMPLATE.format(
        language=output_language,
        chart_name=chart_name,
        analysis_type=analysis_type,
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": image_detail}},
            ],
        }
    )
    return messages
