# phase2_agents/tools.py

import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

from .config import BRAVE_MAX_TOKENS, BRAVE_MAX_URLS

load_dotenv()


BRAVE_API_KEY = os.environ["BRAVE_SEARCH_API_KEY"]
BRAVE_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "search_cache.db"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current or external information and return "
                "relevant source content for answering the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_cache(query_embedding) -> dict | None:
    # return result if there is > 0.95 similarity and None otherwise
    max_sim = 0.0
    max_res = None

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT embedding, response FROM search_cache"
        ).fetchall()

    for embedding_json, response_json in rows:
        embedding = json.loads(embedding_json)
        sim = cosine(embedding, query_embedding)
        if sim > 0.90 and sim > max_sim:
            max_sim = sim
            max_res = json.loads(response_json)
    return max_res

def add_result(query, query_embedding, response: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO search_cache (query, embedding, response)
            VALUES (?, ?, ?)
            """,
            (
                query,
                json.dumps(query_embedding),
                json.dumps(response),
            ),
        )

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                embedding TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

init_db() # runs once at import; safe to repeat because of IF NOT EXISTS

def search_web(query: str) -> dict:
    # check cache, if there's nothin similar in the cache, then run the request
    query_embedding = embed(query)
    result = check_cache(query_embedding)
    if result is not None: # cache hit
        return {
            "query": query,
            "results": result,
        }

    response = requests.get(
        BRAVE_CONTEXT_URL,
        headers={
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json",
        },
        params={ # keep these the same across runs
            "q": query,
            "maximum_number_of_urls": BRAVE_MAX_URLS,
            "maximum_number_of_tokens": BRAVE_MAX_TOKENS,
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for i, item in enumerate(
        data.get("grounding", {}).get("generic", [])
    ):
        results.append(
            {
                "source_id": f"S{i + 1}",
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippets": item.get("snippets", []),
            }
        )
    
    add_result(query, query_embedding, results)

    return {
        "query": query,
        "results": results,
    }


def execute_tool_call(tool_call) -> str:
    tool_name = tool_call["function"]["name"]

    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as exc:
        return json.dumps({
            "error": f"Invalid tool arguments: {exc}"
        })

    try:
        if tool_name == "search_web":
            result = search_web(**arguments)
        else:
            result = {
                "error": f"Unknown tool: {tool_name}"
            }

        return json.dumps(result)

    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "tool": tool_name,
        })








































###### Below here allows the agent to get pages then choose which to load in. Decided against it for V1
# import json
# import os
# from typing import Any

# import requests
# from bs4 import BeautifulSoup

# BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


# DEFAULT_SEARCH_RESULTS = 5
# MAX_SEARCH_RESULTS = 10

# FETCH_TIMEOUT_SECONDS = 10
# MAX_PAGE_CHARS = 20_000

# USER_AGENT = (
#     "Mozilla/5.0 (compatible; WebSearchAgent/0.1; "
#     "+https://github.com/yourname/yourrepo)"
# )


# # ---------------------------------------------------------------------------
# # Tool definitions passed to vLLM / the model
# # ---------------------------------------------------------------------------

# TOOLS = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search_web",
#             "description": (
#                 "Search the public web for information. "
#                 "Use this when the question requires current information, "
#                 "external facts, or sources that are not already in the conversation."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {
#                         "type": "string",
#                         "description": "The web search query.",
#                     },
#                     "num_results": {
#                         "type": "integer",
#                         "description": (
#                             "Number of search results to return. "
#                             "Usually 3 to 5 is enough."
#                         ),
#                         "minimum": 1,
#                         "maximum": MAX_SEARCH_RESULTS,
#                         "default": DEFAULT_SEARCH_RESULTS,
#                     },
#                 },
#                 "required": ["query"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "fetch_page",
#             "description": (
#                 "Fetch and extract readable text from a webpage. "
#                 "Use this after search_web when a search result looks useful "
#                 "and more detail is needed."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "url": {
#                         "type": "string",
#                         "description": "The full HTTP or HTTPS URL to fetch.",
#                     },
#                 },
#                 "required": ["url"],
#             },
#         },
#     },
# ]


# # ---------------------------------------------------------------------------
# # Tool implementations
# # ---------------------------------------------------------------------------

# def search_web(
#     query: str,
#     num_results: int = DEFAULT_SEARCH_RESULTS,
# ) -> dict[str, Any]:
#     """
#     Search the web using Brave Search API.

#     Returns a JSON-serializable dictionary like:

#     {
#         "query": "...",
#         "results": [
#             {
#                 "source_id": "S1",
#                 "title": "...",
#                 "url": "...",
#                 "snippet": "..."
#             }
#         ]
#     }
#     """

#     api_key = os.environ.get("BRAVE_SEARCH_API_KEY")

#     if not api_key:
#         raise RuntimeError(
#             "BRAVE_SEARCH_API_KEY environment variable is not set."
#         )

#     if not query or not query.strip():
#         raise ValueError("query must not be empty")

#     num_results = max(1, min(num_results, MAX_SEARCH_RESULTS))

#     headers = {
#         "Accept": "application/json",
#         "X-Subscription-Token": api_key,
#     }

#     params = {
#         "q": query,
#         "count": num_results,
#         "country": "US",
#         "search_lang": "en",
#     }

#     response = requests.get(
#         BRAVE_SEARCH_URL,
#         headers=headers,
#         params=params,
#         timeout=FETCH_TIMEOUT_SECONDS,
#     )

#     response.raise_for_status()

#     data = response.json()

#     raw_results = data.get("web", {}).get("results", [])

#     results = []

#     for i, result in enumerate(raw_results[:num_results], start=1):
#         results.append(
#             {
#                 "source_id": f"S{i}",
#                 "title": result.get("title", ""),
#                 "url": result.get("url", ""),
#                 "snippet": result.get("description", ""),
#             }
#         )

#     return {
#         "query": query,
#         "results": results,
#     }


# def fetch_page(url: str) -> dict[str, Any]:
#     """
#     Fetch a webpage and extract basic readable text.

#     This intentionally stays simple for the first version of the agent.
#     It will not handle JavaScript-heavy sites particularly well.
#     """

#     if not url.startswith(("http://", "https://")):
#         raise ValueError("URL must start with http:// or https://")

#     headers = {
#         "User-Agent": USER_AGENT,
#         "Accept": "text/html,application/xhtml+xml",
#     }

#     response = requests.get(
#         url,
#         headers=headers,
#         timeout=FETCH_TIMEOUT_SECONDS,
#         allow_redirects=True,
#     )

#     response.raise_for_status()

#     content_type = response.headers.get("content-type", "")

#     if "text/html" not in content_type.lower():
#         return {
#             "url": response.url,
#             "title": "",
#             "text": "",
#             "error": f"Unsupported content type: {content_type}",
#         }

#     soup = BeautifulSoup(response.text, "html.parser")

#     # Remove elements that are usually useless to the model.
#     for tag in soup(
#         [
#             "script",
#             "style",
#             "noscript",
#             "svg",
#             "iframe",
#             "nav",
#             "footer",
#             "form",
#         ]
#     ):
#         tag.decompose()

#     title = ""

#     if soup.title and soup.title.string:
#         title = soup.title.string.strip()

#     text = soup.get_text(separator="\n")

#     # Clean excessive whitespace / blank lines.
#     lines = [
#         line.strip()
#         for line in text.splitlines()
#         if line.strip()
#     ]

#     cleaned_text = "\n".join(lines)

#     # Keep one webpage from blowing up your model context.
#     cleaned_text = cleaned_text[:MAX_PAGE_CHARS]

#     return {
#         "url": response.url,
#         "title": title,
#         "text": cleaned_text,
#     }


# # ---------------------------------------------------------------------------
# # Dispatcher used by agent.py
# # ---------------------------------------------------------------------------

# def execute_tool_call(tool_call) -> str:
#     name = tool_call.function.name
#     arguments = json.loads(tool_call.function.arguments)

#     if name == "search_web":
#         result = search_web(**arguments)

#     elif name == "fetch_page":
#         result = fetch_page(**arguments)

#     else:
#         result = {
#             "error": f"Unknown tool: {name}"
#         }

#     return json.dumps(result)

