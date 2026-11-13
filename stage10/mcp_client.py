"""Stage 10 elective: a minimal MCP client.

MCP (Model Context Protocol) is a standard way to plug outside tools into
any harness. An MCP server is a separate program that offers tools; the
harness starts it, asks what tools it has, and calls them for the model.
The model can't tell an MCP tool from a built-in one.

This client speaks the simplest transport: the server runs as a child
process, and the two exchange JSON-RPC 2.0 messages, one JSON object per
line, over the server's stdin and stdout. Only three requests are needed:

    initialize   say hello and agree on a protocol version
    tools/list   get the tools: name, description, inputSchema (JSON Schema)
    tools/call   run one tool with arguments; get back text content

Servers are listed in mcp.json in the folder the agent starts in:

    {"servers": {"example": {"command": "python", "args": ["mcp_server_example.py"]}}}

Spec: modelcontextprotocol.io
"""

import json
import os
import subprocess

PROTOCOL_VERSION = "2025-06-18"


class MCPError(Exception):
    pass


class MCPServer:
    def __init__(self, name, command, args=(), env=None):
        self.name = name
        self.next_id = 1
        os.makedirs("logs", exist_ok=True)
        self.stderr = open(os.path.join("logs", f"mcp-{name}.log"), "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", bufsize=1,
            env={**os.environ, **(env or {})},
        )
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "learning-harness", "version": "0.1"},
        })
        self.server_info = result.get("serverInfo", {})
        # A "notification" has no id and gets no answer.
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method, params):
        """Send a request and wait for the response with the same id."""
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise MCPError(f"server {self.name!r} stopped (see logs/mcp-{self.name}.log)")
            message = json.loads(line)
            if message.get("id") != request_id:
                continue  # a notification or log message from the server; ignore it
            if "error" in message:
                raise MCPError(f"{self.name}: {message['error'].get('message', message['error'])}")
            return message.get("result", {})

    def list_tools(self):
        return self.request("tools/list", {}).get("tools", [])

    def call_tool(self, tool_name, arguments):
        result = self.request("tools/call", {"name": tool_name, "arguments": arguments})
        # Results are a list of content items; this client only shows text.
        text = "\n".join(item.get("text", f"[{item.get('type')} content]") for item in result.get("content", []))
        return ("Error: " + text) if result.get("isError") else text

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.stderr.close()


def load_tools(config_path="mcp.json"):
    """Start every server in mcp.json and turn its tools into harness tools.
    Returns (tools, servers)."""
    if not os.path.isfile(config_path):
        return [], []
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    tools, servers = [], []
    for name, spec in config.get("servers", {}).items():
        try:
            server = MCPServer(name, spec["command"], spec.get("args", []), spec.get("env"))
            servers.append(server)
            for tool in server.list_tools():
                tools.append(make_tool(server, tool))
        except (OSError, MCPError, ValueError) as error:
            print(f"[MCP server {name!r} failed to start: {error}]")
    return tools, servers


def make_tool(server, tool):
    # Prefix the name with the server's, so two servers can't clash.
    def call(**arguments):
        return server.call_tool(tool["name"], arguments)

    return {
        "name": f"{server.name}__{tool['name']}",
        "kind": "external",  # unknown effects: permissions.py asks first
        "description": tool.get("description", ""),
        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        "function": call,
    }
