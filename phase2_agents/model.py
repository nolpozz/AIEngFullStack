
from openai import OpenAI


MODEL_NAME = "IFM/K2-Horizon-32B" # k2 or whatever we use
BASE_URL = (
    "https://nolpozz--vllm-inference-for-web-search-agent-server-dev."
    "us-east.modal.direct/v1"
)

client = OpenAI( # I'm p sure this is stil how these work
    base_url=BASE_URL,
    api_key="dummy", # still required to pass even tho the server doesn't require authentication
)

def chat(messages, tools=None, temperature = 0.0, max_tokens=1024):
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    
    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message

