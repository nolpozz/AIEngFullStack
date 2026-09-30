
import model

# Might have to modernize this this is just how I remember them from a year ago

MAX_LOOPS = 10



TOOLS = [
    {

    },
    {

    },
]


messages = [ # check that the model we're using uses these same tags
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]


for step in range(MAX_LOOPS):

    response = model.chat(
        messages=messages,
        tools=TOOLS
    )

    messages.append(response)

    if not response.tool_calls:
        return response.content

    for call in response.tool_calls:
        if call.name == "search_web":
            result = search_web(**call.arguments)
        elif call.name == fetch_page:
            reseult = fetch_page(**call.arguments)
        
        # tool logging sent to tracing with metrics
        
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
    
    return MaxStepsExceeded()
