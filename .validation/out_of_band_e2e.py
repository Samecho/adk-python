"""A peer task resumed out-of-band never resumes the RemoteA2aAgent caller.

coordinator (chat LlmAgent, ADK executor)
  --A2A, RemoteA2aAgent(mode="task")--> approver (LlmAgent, ADK executor)

approver's tool needs a human's confirmation, so both tasks pause. The human's
decision is then delivered straight to approver's task (out-of-band from the
coordinator's point of view, A2A spec 7.6.1). approver resumes and completes.
The coordinator is never told: it does not subscribe to, poll, or register a
push config for the peer task (spec 7.6.2), so its own task stays interrupted.
Scripted models: no network, no credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
import warnings

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi
from a2a.server.routes import create_agent_card_routes
from a2a.server.routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentInterface
from fastapi import FastAPI
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from google.protobuf.json_format import MessageToDict
import httpx

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

H = {"A2A-Version": "1.0"}
approver_methods: list[str] = []  # JSON-RPC methods the approver receives
approver_pause: dict = {}
served_runners = []
clients = []
delete_calls = []


def delete_records(table: str) -> dict:
  """Deletes every record in a table. Destructive: needs a human's approval."""
  delete_calls.append(table)
  return {"status": "deleted", "table": table}


class CallOnce(BaseLlm):
  """Calls one tool; once it has the tool's response, answers with text."""

  tool: str
  args: dict = {}
  task_mode: bool = False

  async def generate_content_async(self, llm_request, stream=False):
    fr = llm_request.contents[-1].parts[-1].function_response
    if fr:
      part = (
          types.Part(
              function_call=types.FunctionCall(
                  name="finish_task", args={"result": f"done: {fr.response}"}
              )
          )
          if self.task_mode
          else types.Part(text=f"done: {fr.response}")
      )
    else:
      part = types.Part(
          function_call=types.FunctionCall(name=self.tool, args=self.args)
      )
    yield LlmResponse(content=types.Content(role="model", parts=[part]))


def card(name):
  return AgentCard(
      name=name,
      description=name,
      version="1",
      capabilities=AgentCapabilities(streaming=True),
      default_input_modes=["text/plain"],
      default_output_modes=["text/plain"],
      supported_interfaces=[
          AgentInterface(
              url=f"http://{name}.test/a2a", protocol_binding="JSONRPC"
          )
      ],
  )


def record_pause(event):
  """Remember the approver's paused task and its confirmation call id."""
  status = getattr(event, "status", None)
  if status is None or not status.HasField("message"):
    return
  for p in status.message.parts:
    if p.HasField("data"):
      d = MessageToDict(p.data)
      if d.get("name") == "adk_request_confirmation" and "args" in d:
        approver_pause.update(
            task_id=event.task_id,
            context_id=event.context_id,
            confirmation_id=d["id"],
        )


class Recording(httpx.AsyncBaseTransport):
  """Log each JSON-RPC method sent to the approver, then pass it through."""

  def __init__(self, inner):
    self.inner = inner

  async def handle_async_request(self, request):
    if request.method == "POST":
      approver_methods.append(json.loads(request.content).get("method"))
    return await self.inner.handle_async_request(request)


def serve(agent, on_event=None):
  """Serve an ADK agent with ADK's own A2aAgentExecutor, in process."""
  runner = Runner(
      app_name=agent.name,
      agent=agent,
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )
  served_runners.append(runner)
  c = card(agent.name)
  handler = DefaultRequestHandler(
      A2aAgentExecutor(runner=runner, force_new_version="--v2" in sys.argv),
      InMemoryTaskStore(),
      c,
  )
  if on_event:
    orig = handler.agent_executor.execute

    async def execute(context, queue):
      orig_enqueue = queue.enqueue_event

      async def enqueue(e):
        on_event(e)
        await orig_enqueue(e)

      queue.enqueue_event = enqueue
      await orig(context, queue)

    handler.agent_executor.execute = execute
  app = FastAPI()
  add_a2a_routes_to_fastapi(
      app,
      agent_card_routes=create_agent_card_routes(c),
      jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/a2a"),
  )
  return httpx.ASGITransport(app=app)


async def rpc(client, name, method, params):
  body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
  print("rpc", name, method, flush=True)
  r = await asyncio.wait_for(
      client.post(f"http://{name}.test/a2a", json=body, headers=H), 15
  )
  res = r.json()["result"]
  return res.get("task", res)


def client(transport):
  value = httpx.AsyncClient(transport=transport)
  clients.append(value)
  return value


async def main():
  approver_asgi = serve(
      LlmAgent(
          name="approver",
          mode="task",
          model=CallOnce(
              model="fake",
              tool="delete_records",
              args={"table": "t"},
              task_mode=True,
          ),
          tools=[FunctionTool(delete_records, require_confirmation=True)],
      ),
      record_pause,
  )
  approver_t = Recording(approver_asgi)  # only the coordinator's traffic
  coordinator_t = serve(
      LlmAgent(
          name="coordinator",
          model=CallOnce(model="fake", tool="approver"),
          sub_agents=[
              RemoteA2aAgent(
                  name="approver",
                  description="approver",
                  agent_card=card("approver"),
                  mode="task",
                  httpx_client=client(approver_t),
              )
          ],
      )
  )
  to_approver = client(approver_asgi)  # the human's side, unrecorded
  to_coordinator = client(coordinator_t)

  # 1. The user asks the coordinator; the approver pauses for a human.
  msg = {
      "messageId": uuid.uuid4().hex,
      "role": "ROLE_USER",
      "parts": [{"text": "go"}],
  }
  coord = await rpc(
      to_coordinator, "coordinator", "SendMessage", {"message": msg}
  )
  print("1. coordinator task:", coord["status"]["state"])
  print(
      "   approver  task:",
      (
          await rpc(
              to_approver,
              "approver",
              "GetTask",
              {"id": approver_pause["task_id"]},
          )
      )["status"]["state"],
  )
  before = list(approver_methods)

  # 2. The human's decision goes straight to the approver's task.
  decision = {
      "messageId": uuid.uuid4().hex,
      "role": "ROLE_USER",
      "taskId": approver_pause["task_id"],
      "contextId": approver_pause["context_id"],
      "parts": [{
          "data": {
              "id": approver_pause["confirmation_id"],
              "name": "adk_request_confirmation",
              "response": {"confirmed": True},
          },
          "metadata": {"adk_type": "function_response"},
      }],
  }
  done = await rpc(
      to_approver, "approver", "SendMessage", {"message": decision}
  )
  texts = [p.get("text") for a in done.get("artifacts", []) for p in a["parts"]]
  print("2. approver  task:", done["status"]["state"], texts)

  # 3. Later: the coordinator has not noticed.
  await asyncio.sleep(2)
  coord = await rpc(
      to_coordinator, "coordinator", "GetTask", {"id": coord["id"]}
  )
  print("3. coordinator task:", coord["status"]["state"], "(after 2 s)")
  print("   coordinator -> approver before the decision:", before)
  print("   coordinator -> approver after it:", approver_methods[len(before) :])

  followup = {
      "messageId": uuid.uuid4().hex,
      "role": "ROLE_USER",
      "taskId": coord["id"],
      "contextId": coord["contextId"],
      "parts": decision["parts"],
  }
  resumed = await rpc(
      to_coordinator, "coordinator", "SendMessage", {"message": followup}
  )
  print("4. coordinator next turn:", resumed["status"]["state"])
  print(
      "   output:",
      [p.get("text") for a in resumed.get("artifacts", []) for p in a["parts"]],
  )
  print("   coordinator -> approver methods:", approver_methods)
  outputs = [
      p.get("text", "")
      for a in resumed.get("artifacts", [])
      for p in a["parts"]
  ]
  assert resumed["status"]["state"] == "TASK_STATE_COMPLETED"
  assert any(
      "deleted" in text and "done: {'result':" in text for text in outputs
  ), outputs
  assert not any("done: {'error':" in text for text in outputs), outputs
  assert approver_methods == [
      "SendStreamingMessage",
      "GetTask",
  ], approver_methods
  assert delete_calls == ["t"], delete_calls
  print("PASS: completed result recovered; no repeated send or tool execution")


async def run():
  try:
    await asyncio.wait_for(main(), 60)
  finally:
    for runner in reversed(served_runners):
      await runner.close()
    for value in clients:
      await value.aclose()


asyncio.run(run())
