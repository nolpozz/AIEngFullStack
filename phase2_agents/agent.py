import time

from .config import MAX_STEPS
from .model import chat
from .tools import TOOLS, execute_tool_call
from tracing.tracer import Tracer


SYSTEM_PROMPT = """
You are a web research assistant.

Use search_web when the user's question requires current,
external, or web-based information.

When using search results:
- Prefer authoritative sources.
- Base factual claims on retrieved evidence.
- Cite sources using their source IDs, like [S1] or [S2].
- Do not search unnecessarily.
- If the retrieved information is insufficient, search again with
  a better query.
"""


def run_agent(question: str, max_steps: int = MAX_STEPS) -> str:
    tracer = Tracer(question)
    status = "error" # overwritten if we finish cleanly or run out of steps

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    try:
        for step in range(max_steps):
            started_at = time.time()

            try:
                message, stats = chat(
                    messages=messages,
                    tools=TOOLS,
                )
            except Exception as e:
                tracer.log_model_error(step, started_at, e)
                continue # retry

            tracer.log_model_call(
                call_index=step,
                timestamp=started_at,
                **stats,
            )

            messages.append(message)

            if not message.get("tool_calls"):
                status = "ok"
                return message["content"] or ""

            for tool_call in message["tool_calls"]:
                tool_started_at = time.time()
                tool_start = time.perf_counter()

                result = execute_tool_call(tool_call)

                tool_latency_ms = (
                    time.perf_counter() - tool_start
                ) * 1000

                tracer.log_tool_call(
                    step=step,
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["arguments"],
                    latency_ms=tool_latency_ms,
                    timestamp=tool_started_at,
                    result_chars=len(result),
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

        status = "max_steps"
        raise RuntimeError(
            f"Agent exceeded max_steps={max_steps}"
        )

    finally:
        tracer.finish(status)