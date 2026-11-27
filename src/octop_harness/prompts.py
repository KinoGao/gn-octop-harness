"""Runtime execution-discipline prompt shared across all routed models.

Upstream ``deepagents`` only attaches behavioral prompt profiles for a small
fixed set of frontier models at graph compile time. ``octop-harness`` routes
requests dynamically across many providers and models, so it appends this
compact, cache-stable execution discipline block to the compiled system prompt.
"""

from __future__ import annotations

_DISCIPLINE_ZH = "\n".join(
    (
        "## 执行准则",
        "",
        "- **先查证再结论**: 涉及文件、代码、数据或外部系统状态时, 先用工具读取核实, 不臆测未查看的内容。",
        "- **独立操作并行发起**: 无先后依赖的多个读取或搜索调用应在同一轮并行发出; 仅在后续参数依赖前序结果时串行。",
        (
            "- **理解 -> 执行 -> 核验**: 修改文件或执行操作后, "
            "用工具检查产出或运行测试确认生效, 不要把首次尝试当成最终完成。"
        ),
        (
            "- **遇错先分析再换路**: 工具报错或返回空结果时, 先分析根因并调整策略或缩小范围, "
            "不要盲目重复同样调用, 也不凭空补全缺失证据。"
        ),
    )
)

_DISCIPLINE_EN = "\n".join(
    (
        "## Execution Discipline",
        "",
        (
            "- **Investigate before answering**: When a task depends on files, code, data, or external system state, "
            "observe it with tools first; never speculate about uninspected content."
        ),
        (
            "- **Batch independent calls**: Issue independent read/search tool calls in parallel in the same turn; "
            "run sequentially only when a call depends on a prior result."
        ),
        (
            "- **Understand -> Act -> Verify**: Gather enough context before acting, "
            "then verify edits or outputs with tools before claiming completion."
        ),
        (
            "- **Reflect on failures**: When a tool fails or returns empty evidence, diagnose why and adjust strategy "
            "instead of repeating the same call or inventing missing facts."
        ),
    )
)


def render_execution_discipline_prompt(*, language: str = "zh") -> str:
    """Return the cache-stable execution-discipline section for the system prompt."""
    return _DISCIPLINE_EN if language == "en" else _DISCIPLINE_ZH


__all__ = ["render_execution_discipline_prompt"]
