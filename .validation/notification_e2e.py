from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from google.adk.agents.llm_agent import LlmAgent
from google.adk.dependencies._mcp import Context as McpContext
from google.adk.dependencies._mcp import FastMCP
from google.adk.dependencies._mcp import StdioServerParameters
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types


class NotificationModel(BaseLlm):
    model: str = "notification-test-model"

    async def generate_content_async(self, llm_request, stream=False):
        has_response = any(
            part.function_response
            for content in llm_request.contents
            for part in content.parts or []
        )
        part = (
            types.Part(text="done")
            if has_response
            else types.Part(
                function_call=types.FunctionCall(name="notify", args={})
            )
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def serve(transport, port):
    server = FastMCP("notification-e2e")

    @server.tool()
    async def notify(ctx: McpContext) -> str:
        await ctx.session.send_tool_list_changed()
        return "sent"

    if transport == "stdio":
        server.run()
    else:
        import uvicorn

        app = (
            server.sse_app()
            if transport == "sse"
            else server.streamable_http_app()
        )
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


async def runner_case(params, with_handler):
    messages = []
    received = asyncio.Event()

    async def handler(message):
        messages.append(getattr(message, "root", message).method)
        received.set()

    options = {"message_handler": handler} if with_handler else {}
    toolset = McpToolset(connection_params=params, **options)
    agent = LlmAgent(name="notification_agent", model=NotificationModel(), tools=[toolset])
    runner = InMemoryRunner(agent=agent, app_name="notification_e2e")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test"
    )
    try:
        events = [
            event
            async for event in runner.run_async(
                user_id="test",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="notify me")]
                ),
            )
        ]
        if with_handler:
            await asyncio.wait_for(received.wait(), timeout=5)
        responses = [response for event in events for response in event.get_function_responses()]
        assert len(responses) == 1, responses
        assert responses[0].response["content"][0]["text"] == "sent", responses
        assert any(
            part.text == "done"
            for event in events
            if event.content
            for part in event.content.parts or []
        )
        assert messages == (["notifications/tools/list_changed"] if with_handler else []), messages
    finally:
        await runner.close()


async def pool_case(params):
    messages = []
    received = asyncio.Event()

    async def handler(message):
        messages.append(getattr(message, "root", message).method)
        received.set()

    manager = MCPSessionManager(params, message_handler=handler)
    try:
        first = await manager.create_session()
        for _ in range(2):
            received.clear()
            session = await manager.create_session()
            assert session is first
            await session.call_tool("notify", arguments={})
            await asyncio.wait_for(received.wait(), timeout=5)
        await manager.close()
        received.clear()
        replacement = await manager.create_session()
        assert replacement is not first
        await replacement.call_tool("notify", arguments={})
        await asyncio.wait_for(received.wait(), timeout=5)
        assert messages == ["notifications/tools/list_changed"] * 3, messages
    finally:
        await manager.close()


async def main():
    os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
    results = []
    script = str(Path(__file__).resolve())
    for transport in ("stdio", "sse", "streamable_http"):
        process = None
        with tempfile.TemporaryFile(mode="w+b") as log:
            if transport == "stdio":
                params = StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=sys.executable,
                        args=[script, "server", transport, "0"],
                        env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                    ),
                    timeout=20,
                )
            else:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                process = subprocess.Popen(
                    [sys.executable, script, "server", transport, str(port)],
                    stdout=log,
                    stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for _ in range(200):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read().decode(errors="replace"))
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                            break
                    except OSError:
                        await asyncio.sleep(0.1)
                else:
                    process.terminate()
                    raise TimeoutError("MCP test server did not start")
                cls = SseConnectionParams if transport == "sse" else StreamableHTTPConnectionParams
                path = "sse" if transport == "sse" else "mcp"
                params = cls(url=f"http://127.0.0.1:{port}/{path}", timeout=20)
            try:
                for with_handler in (False, True):
                    await asyncio.wait_for(runner_case(params, with_handler), timeout=60)
                    results.append({"transport": transport, "case": "runner", "handler": with_handler, "passed": True})
                await asyncio.wait_for(pool_case(params), timeout=60)
                results.append({"transport": transport, "case": "reuse_and_reconnect", "passed": True})
            finally:
                if process is not None:
                    process.terminate()
                    process.wait(timeout=10)
    import google.adk

    print(json.dumps({"mcp_version": importlib.metadata.version("mcp"), "adk_path": str(Path(google.adk.__file__).resolve()), "passed": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        serve(sys.argv[2], int(sys.argv[3]))
    else:
        asyncio.run(main())
