# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path)
parser.add_argument("--result", required=True, type=Path)
options = parser.parse_args()
if options.source:
  sys.path.insert(0, str(options.source.resolve()))

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.environment_simulation.environment_simulation_config import EnvironmentSimulationConfig
from google.adk.tools.environment_simulation.environment_simulation_engine import EnvironmentSimulationEngine
from google.adk.tools.environment_simulation.environment_simulation_plugin import EnvironmentSimulationPlugin
from google.genai import types


class ScriptedModel(BaseLlm):
  model: str = "mock"
  calls: int = 0

  async def generate_content_async(self, llm_request, stream=False):
    self.calls += 1
    if self.calls == 1:
      part = types.Part.from_function_call(name="fetch_data", args={"key": "x"})
    else:
      part = types.Part.from_text(text="done")
    yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def run_scenario(name, injection, expected_response, expected_calls):
  tool_calls = []

  def fetch_data(key: str) -> dict[str, bool]:
    tool_calls.append(key)
    return {"live": True}

  config = EnvironmentSimulationConfig.model_validate_json(
      json.dumps({
          "tool_simulation_configs": [
              {"tool_name": "fetch_data", "injection_configs": [injection]}
          ]
      })
  )
  model = ScriptedModel()
  agent = LlmAgent(name="test_agent", model=model, tools=[fetch_data])
  runner = Runner(
      app=App(
          name="simulation_e2e",
          root_agent=agent,
          plugins=[
              EnvironmentSimulationPlugin(EnvironmentSimulationEngine(config))
          ],
      ),
      session_service=InMemorySessionService(),
  )
  try:
    session = await runner.session_service.create_session(
        app_name="simulation_e2e", user_id="e2e"
    )
    events = [
        event
        async for event in runner.run_async(
            user_id="e2e",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="fetch data")]
            ),
        )
    ]
    responses = [
        result for event in events for result in event.get_function_responses()
    ]
    assert len(responses) == 1, responses
    assert responses[0].response == expected_response, responses
    assert tool_calls == expected_calls, tool_calls
    assert model.calls == 2, model.calls
    assert events[-1].content.parts[0].text == "done", events[-1]
    return {
        "scenario": name,
        "tool_response": responses[0].response,
        "real_tool_calls": tool_calls,
        "model_calls": model.calls,
        "passed": True,
    }
  finally:
    await runner.close()


async def main():
  cases = [
      ("empty_response", {"injected_response": {}}, {}, []),
      (
          "nonempty_response",
          {"injected_response": {"status": "ok"}},
          {"status": "ok"},
          [],
      ),
      (
          "error",
          {
              "injected_error": {
                  "injected_http_error_code": 404,
                  "error_message": "not found",
              }
          },
          {"error_code": 404, "error_message": "not found"},
          [],
      ),
      (
          "unmatched_args",
          {"injected_response": {}, "match_args": {"key": "other"}},
          {"live": True},
          ["x"],
      ),
  ]
  results = [await run_scenario(*case) for case in cases]
  import google.adk

  report = {
      "python": sys.version,
      "package_version": importlib.metadata.version("google-adk"),
      "package_path": google.adk.__file__,
      "scenarios": results,
  }
  options.result.write_text(
      json.dumps(report, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, indent=2))


asyncio.run(main())
