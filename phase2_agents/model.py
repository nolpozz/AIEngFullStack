
from openai import openai


MODEL_NAME = "" # k2 or whatever we use
BASE_URL = "" # modal url


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
    
    response = client.chat.completion.create(**kwargs)

    return response.choices[0].message

