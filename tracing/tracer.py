# tracing/tracer.py

import json
import time
import uuid
from pathlib import Path


TRACE_FILE = Path(__file__).parent / "traces.jsonl"


class Tracer:
    def __init__(self):
        self.run_id = str(uuid.uuid4())
        self.start_time = time.perf_counter()
        self.seen_tool_calls = set()

        self.model_calls = 0
        self.tool_calls = 0
        self.repeated_tool_calls = 0

    def log(self, event: dict) -> None:
        event["run_id"] = self.run_id

        with TRACE_FILE.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def log_model_call(
        self,
        call_index: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        timestamp: float,
    ) -> None:
        self.model_calls += 1

        self.log({
            "type": "model",
            "call_index": call_index,
            "timestamp": timestamp,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        })

    def log_tool_call(
        self,
        name: str,
        arguments: str,
        latency_ms: float,
    ) -> None:
        self.tool_calls += 1

        signature = (name, arguments)
        repeated = signature in self.seen_tool_calls

        if repeated:
            self.repeated_tool_calls += 1

        self.seen_tool_calls.add(signature)

        self.log({
            "type": "tool",
            "tool": name,
            "arguments": arguments,
            "latency_ms": latency_ms,
            "repeated": repeated,
        })

    def finish(self) -> None:
        self.log({
            "type": "run_end",
            "duration_ms": (
                time.perf_counter() - self.start_time
            ) * 1000,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "repeated_tool_calls": self.repeated_tool_calls,
        })