# tracing/tracer.py

import json
import threading
import time
import uuid
from pathlib import Path

from phase2_agents.config import CONFIG, CONFIG_ID


TRACE_FILE = Path(__file__).parent / "traces.jsonl"
_write_lock = threading.Lock() # one writer at a time across agent threads


class Tracer:
    def __init__(self, question: str):
        self.run_id = str(uuid.uuid4())
        self.start_time = time.perf_counter()
        self.seen_tool_calls = set()

        self.model_calls = 0
        self.tool_calls = 0
        self.repeated_tool_calls = 0

        self.log({"type": "run_start", "question": question, "config": CONFIG})

    def log(self, event: dict) -> None:
        event["run_id"] = self.run_id
        event["config_id"] = CONFIG_ID
        event.setdefault("timestamp", time.time()) # wall clock; joins with the vLLM metrics scrape

        line = json.dumps(event) + "\n"
        with _write_lock, TRACE_FILE.open("a") as f:
            f.write(line)

    def log_model_call(self, call_index: int, timestamp: float, **stats) -> None:
        # stats comes straight from model.chat(): tokens, latency, ttft, decode, finish_reason
        self.model_calls += 1

        self.log({
            "type": "model",
            "call_index": call_index,
            "timestamp": timestamp,
            **stats,
        })

    def log_model_error(self, call_index: int, timestamp: float, error: Exception) -> None:
        self.log({
            "type": "model_error",
            "call_index": call_index,
            "timestamp": timestamp,
            "error": f"{type(error).__name__}: {error}",
        })

    def log_tool_call(
        self,
        step: int,
        name: str,
        arguments: str,
        latency_ms: float,
        timestamp: float,
        result_chars: int,
    ) -> None:
        self.tool_calls += 1

        signature = (name, arguments)
        repeated = signature in self.seen_tool_calls

        if repeated:
            self.repeated_tool_calls += 1

        self.seen_tool_calls.add(signature)

        self.log({
            "type": "tool",
            "step": step,
            "timestamp": timestamp,
            "tool": name,
            "arguments": arguments,
            "latency_ms": latency_ms,
            "result_chars": result_chars,
            "repeated": repeated,
        })

    def finish(self, status: str) -> None:
        self.log({
            "type": "run_end",
            "status": status, # ok | max_steps | error
            "duration_ms": (
                time.perf_counter() - self.start_time
            ) * 1000,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "repeated_tool_calls": self.repeated_tool_calls,
        })