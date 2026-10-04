# Phase 2 — Web Search Agent

This directory contains a minimal web-search agent built on top of the self-hosted vLLM inference service from Phase 1.

The goal of this phase is to build a functional agent while keeping the implementation small enough to understand every part of the control flow.

The agent uses:

- `IFM/K2-Horizon-32B`
- vLLM
- Modal
- the OpenAI-compatible chat-completions API
- Brave Search's LLM Context endpoint
- native model tool calling

No agent framework is used.

---

# Architecture

```text
User
 ↓
cli.py
 ↓
agent.py
 ↓
model.py
 ↓
vLLM server
 ↓
K2
 ↓
structured tool call
 ↓
agent.py
 ↓
tools.py
 ↓
Brave LLM Context API
 ↓
tool result
 ↓
agent.py
 ↓
model.py
 ↓
K2
 ↓
final answer
```

The model does not directly execute tools.

It only produces a structured request describing which tool should be called and with which arguments.

The Python agent executes the requested tool and sends the result back to the model.

---

# Files

```text
phase2_agents/
├── agent.py
├── cli.py
├── config.py
├── model.py
├── tools.py
└── README.md
```

---

## `model.py`

`model.py` is the inference client.

It is responsible for communicating with the self-hosted vLLM endpoint.

It accepts:

- conversation messages
- tool definitions
- generation parameters

and returns the model's response.

Conceptually:

```text
messages + tools
      ↓
model.py
      ↓
POST /v1/chat/completions
      ↓
vLLM
      ↓
K2 response
```

The application uses the official OpenAI Python client because vLLM exposes an OpenAI-compatible API.

The inference is still performed entirely by the self-hosted model.

---

## `tools.py`

`tools.py` defines the capabilities available to the agent.

For v1, the agent has one tool:

```text
search_web(query)
```

This file contains both:

1. the JSON tool definition shown to the model
2. the actual Python implementation of the tool

Keeping these together helps ensure that the model-visible schema stays synchronized with the underlying function.

For example:

```text
tool schema:
search_web(query: string)

         ↕

Python implementation:
def search_web(query: str)
```

---

## `agent.py`

`agent.py` contains the control loop.

Its responsibility is to:

1. construct the initial conversation
2. call the model
3. inspect the response for tool calls
4. execute requested tools
5. append tool results
6. call the model again
7. stop when the model returns a normal answer

The basic loop is:

```python
for step in range(max_steps):
    message, stats = chat(messages, tools=TOOLS)
    tracer.log_model_call(step, ..., **stats)

    messages.append(message)

    if not message.get("tool_calls"):
        return message["content"]

    for tool_call in message["tool_calls"]:
        result = execute_tool_call(tool_call)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result,
        })
```

`chat()` streams the response and returns the assistant message as a dict plus `stats` (token counts, TTFT, decode time), which go to the tracer.

The maximum number of steps prevents a model from searching indefinitely.

---

## `cli.py`

`cli.py` is a minimal command-line interface.

It allows the agent to be called as:

```bash
python -m phase2_agents.cli "What is the latest version of vLLM?"
```

or interactively:

```bash
python -m phase2_agents.cli
```

The CLI intentionally contains very little logic.

Its job is only to:

- collect the user's question
- pass it to `run_agent`
- print the result

---

# Web Search

The first version uses Brave Search's LLM Context endpoint.

The agent exposes:

```text
search_web(query)
```

and Brave performs the lower-level retrieval work.

The flow is:

```text
search query
     ↓
Brave Search
     ↓
relevant web sources
     ↓
extracted grounding snippets
     ↓
agent context
```

This avoids implementing page downloading, HTML extraction, ranking, and chunking during the first iteration.

A later version may replace this with separate lower-level tools such as:

```text
search_web(query)
fetch_page(url)
```

which would give the model more control over the retrieval process.

---

# Setup

## Install dependencies

From the project environment:

```bash
pip install openai requests
```

---

## Configure Brave Search

Set the Brave Search API key:

```bash
export BRAVE_SEARCH_API_KEY="your-api-key"
```

`tools.py` reads this environment variable when executing web searches.

---

## Start the inference server

Start the Phase 1 vLLM server on Modal.

For example:

```bash
modal serve phase1_serving/modalForvLLM.py
```

Modal will return the URL for the deployed service.

---

## Configure `model.py`

Set the vLLM base URL:

```python
BASE_URL = "https://YOUR-MODAL-URL.modal.direct/v1"
```

The model name should match the name used by the vLLM server:

```python
MODEL_NAME = "IFM/K2-Horizon-32B"
```

---

# Running the Agent

From the repository root:

```bash
python -m phase2_agents.cli \
    "What is the latest stable release of vLLM?"
```

The first inference request includes:

- the system prompt
- the user's question
- the `search_web` tool definition

If K2 determines that web search is needed, it may return a structured tool call such as:

```text
search_web(
    query="latest stable release of vLLM"
)
```

The application then:

```text
parses tool arguments
      ↓
calls Brave
      ↓
receives grounding context
      ↓
serializes result
      ↓
adds a tool message
      ↓
calls K2 again
```

K2 can either request another search or produce the final answer.

---

# Current Tool

The initial tool schema is conceptually:

```json
{
  "name": "search_web",
  "parameters": {
    "query": "string"
  }
}
```

The Python function sends the query to:

```text
https://api.search.brave.com/res/v1/llm/context
```

and returns normalized source information containing:

```text
source ID
title
URL
retrieved snippets
```

These results are passed back into the model context.

---

# Why Only One Tool?

The first version is intentionally minimal.

The goal is to verify the complete loop:

```text
LLM
 ↓
tool decision
 ↓
external API
 ↓
tool observation
 ↓
LLM
 ↓
answer
```

before adding complexity.

This makes it much easier to debug:

- tool-call formatting
- model behavior
- inference failures
- API failures
- malformed arguments
- context growth
- termination behavior

---

# Current Limitations

The v1 agent does not yet include:

- browser automation
- direct webpage fetching
- memory
- persistent conversations
- streaming
- parallel tool execution
- multiple agents
- structured tracing
- automated evaluation
- retry policies
- advanced source filtering

These are intentionally deferred.

---

# Next Step — Tracing

The next major addition should be tracing rather than additional tools.

Each model invocation should eventually record information such as:

```json
{
  "type": "model",
  "input_tokens": 1240,
  "output_tokens": 38,
  "latency_ms": 720,
  "ttft_ms": 95
}
```

and each tool call:

```json
{
  "type": "tool",
  "tool": "search_web",
  "latency_ms": 310
}
```

A complete trajectory could then be stored as JSONL.

This creates the data needed for both agent evaluation and inference workload analysis.

---

# Evaluation Plan

The first custom eval set should remain small.

Useful categories include:

### Requires search

Questions whose answers depend on current external information.

Expected behavior:

```text
model
 ↓
search
 ↓
model
 ↓
answer
```

### Does not require search

For example:

```text
What is 7 * 8?
```

Expected behavior:

```text
model
 ↓
answer
```

This checks whether the model is overusing its tool.

### Search reformulation

Questions where the first query may not retrieve sufficient context.

Expected behavior:

```text
model
 ↓
search
 ↓
model
 ↓
better search
 ↓
model
 ↓
answer
```

### Grounding

Check whether the final answer is actually supported by retrieved source content.

---

# Why This Agent Matters for the Larger Project

The agent is not only an application demo.

It also creates a realistic inference workload.

A search trajectory may generate:

```text
call 1:
small input
short tool-call output

call 2:
larger input containing search context
short output

call 3:
larger context
long final answer
```

Different users can also have several of these trajectories running concurrently.

That creates variable:

- prompt lengths
- output lengths
- KV-cache lifetimes
- prefill loads
- decode loads
- request arrival times
- inter-request dependencies

These traces can eventually become the workload used to profile and optimize the inference service.

---

# Phase Goal

The success criterion for this phase is simple:

```text
CLI question
    ↓
self-hosted K2
    ↓
K2 requests web search
    ↓
Brave returns grounding context
    ↓
K2 consumes that context
    ↓
grounded final answer
```

Once that loop works reliably, the next focus is evaluation and tracing rather than adding more agent complexity.