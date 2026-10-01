import time

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


def run_agent(question: str, max_steps: int = 6) -> str:
    tracer = Tracer()

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
            start = time.perf_counter()

            response = chat(
                messages=messages,
                tools=TOOLS,
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            usage = response.usage

            tracer.log_model_call(
                call_index=step,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=latency_ms,
                timestamp=started_at,
            )

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                tool_start = time.perf_counter()

                result = execute_tool_call(tool_call)

                tool_latency_ms = (
                    time.perf_counter() - tool_start
                ) * 1000

                tracer.log_tool_call(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    latency_ms=tool_latency_ms,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        raise RuntimeError(
            f"Agent exceeded max_steps={max_steps}"
        )

    finally:
        tracer.finish()