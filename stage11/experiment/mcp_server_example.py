"""A tiny MCP server, to see the protocol from the other side.

The harness starts this program and talks to it over stdin/stdout, one JSON
message per line (see mcp_client.py). It offers two tools. Anything printed
to stderr ends up in logs/mcp-example.log.
"""

import json
import sys
import time

TOOLS = [
    {
        "name": "word_count",
        "description": "Count the words, lines and characters in a piece of text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to count."}},
            "required": ["text"],
        },
    },
    {
        "name": "current_time",
        "description": "The current date and time on this computer.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call(name, arguments):
    if name == "word_count":
        text = arguments.get("text", "")
        return f"{len(text.split())} words, {len(text.splitlines())} lines, {len(text)} characters"
    if name == "current_time":
        return time.strftime("%A %d %B %Y, %H:%M:%S")
    raise ValueError(f"unknown tool {name}")


def reply(request_id, result):
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    print(f"received {method}", file=sys.stderr, flush=True)
    if "id" not in message:
        continue  # a notification: nothing to answer
    if method == "initialize":
        reply(message["id"], {
            "protocolVersion": message["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "example", "version": "0.1"},
        })
    elif method == "tools/list":
        reply(message["id"], {"tools": TOOLS})
    elif method == "tools/call":
        params = message["params"]
        try:
            text, is_error = call(params["name"], params.get("arguments", {})), False
        except Exception as error:
            text, is_error = str(error), True
        reply(message["id"], {"content": [{"type": "text", "text": text}], "isError": is_error})
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                          "error": {"code": -32601, "message": f"unknown method {method}"}}), flush=True)
