import time

from openai import OpenAI

from .config import (
    BASE_URL,
    MAX_TOKENS,
    MODEL_NAME,
    REASONING_EFFORT,
    TEMPERATURE,
    TOP_P,
)

client = OpenAI( # I'm p sure this is stil how these work
    base_url=BASE_URL,
    api_key="dummy", # still required to pass even tho the server doesn't require authentication
    max_retries=0, # handled in the agent loop
)

def chat(messages, tools=None):
    # Streams one call. Returns (message, stats): message is a dict ready to
    # append to `messages`; stats is the timing/token numbers for the tracer.
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True}, # final chunk carries token counts
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": REASONING_EFFORT}},
    }

    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    t_start = time.perf_counter()
    t_first_any = None      # first reasoning OR content OR tool-call token
    t_first_visible = None  # first content OR tool-call token

    content = ""
    tool_calls = {}  # index -> tool call being assembled from fragments
    usage = None
    finish_reason = None

    for chunk in client.chat.completions.create(**kwargs):
        now = time.perf_counter()

        if chunk.usage: # final chunk: usage set, choices empty
            usage = chunk.usage
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None) # vLLM's extra field

        if (reasoning or delta.content or delta.tool_calls) and t_first_any is None:
            t_first_any = now
        if (delta.content or delta.tool_calls) and t_first_visible is None:
            t_first_visible = now

        if delta.content:
            content += delta.content

        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(tc.index, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["function"]["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["function"]["arguments"] += tc.function.arguments # arrives in pieces

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    t_end = time.perf_counter()

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    stats = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "latency_ms": (t_end - t_start) * 1000,
        "ttft_any_ms": None if t_first_any is None else (t_first_any - t_start) * 1000,
        "ttft_visible_ms": None if t_first_visible is None else (t_first_visible - t_start) * 1000,
        "decode_ms": None if t_first_any is None else (t_end - t_first_any) * 1000,
        "finish_reason": finish_reason, # "length" = hit max_tokens
    }
    return message, stats

