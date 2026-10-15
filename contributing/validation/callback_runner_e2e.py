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
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
parser.add_argument('--result', required=True, type=Path)
args = parser.parse_args()
if args.source:
  sys.path.insert(0, str(args.source.resolve()))
os.environ['ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN'] = '1'

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.telemetry.context import TelemetryConfig
from google.adk.utils import _callback_pipeline
from google.genai import types

loaded = Path(_callback_pipeline.__file__).resolve()
if args.source:
  assert loaded.is_relative_to(args.source.resolve()), loaded
else:
  assert 'site-packages' in loaded.parts, loaded
  assert not any(p.startswith('adk-2354') for p in loaded.parts), loaded

observed = []


class ScriptedModel(BaseLlm):

  async def generate_content_async(self, llm_request, stream=False):
    has_result = any(
        part.function_response
        for content in llm_request.contents
        for part in content.parts or []
    )
    part = (
        types.Part(text='done')
        if has_result
        else types.Part.from_function_call(name='add_one', args={'value': 1})
    )
    yield LlmResponse(content=types.Content(role='model', parts=[part]))


def add_one(value: int) -> dict[str, int]:
  return {'value': value + 1}


def record(name):
  observed.append((name, trace.get_current_span().name))


def before_agent(callback_context):
  record('before_agent')


async def after_agent(callback_context):
  record('after_agent')


async def before_model(callback_context, llm_request):
  record('before_model')


def after_model(callback_context, llm_response):
  record('after_model')


def before_tool(tool, args, tool_context):
  record('before_tool')


async def after_tool(tool, args, tool_context, tool_response):
  record('after_tool')


async def one_case(label, env, override):
  exporter.clear()
  observed.clear()
  if env is None:
    os.environ.pop('ADK_EXPERIMENTAL_TELEMETRY', None)
  else:
    os.environ['ADK_EXPERIMENTAL_TELEMETRY'] = env
  config = RunConfig(
      telemetry=TelemetryConfig(adk_experimental_telemetry_opt_in=override)
  )
  agent = LlmAgent(
      name='guardrail_agent',
      model=ScriptedModel(model='scripted'),
      tools=[add_one],
      before_agent_callback=before_agent,
      after_agent_callback=after_agent,
      before_model_callback=before_model,
      after_model_callback=after_model,
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  service = InMemorySessionService()
  async with Runner(
      app_name='callback_probe',
      agent=agent,
      session_service=service,
      auto_create_session=True,
  ) as runner:
    events = [
        event
        async for event in runner.run_async(
            user_id='user',
            session_id='session',
            new_message=types.Content(
                role='user', parts=[types.Part(text='private-input')]
            ),
            run_config=config,
        )
    ]
  assert events[-1].content.parts[0].text == 'done'
  session = await service.get_session(
      app_name='callback_probe', user_id='user', session_id='session'
  )
  assert session.events[-1].content.parts[0].text == 'done'
  provider.force_flush()
  spans = exporter.get_finished_spans()
  callbacks = [s for s in spans if s.name.startswith('execute_callback ')]
  enabled = override if override is not None else env == 'true'
  assert len(observed) == 8, observed
  assert len(callbacks) == (8 if enabled else 0), [s.name for s in spans]
  parents = {s.context.span_id: s.name for s in spans}
  if enabled:
    for name, current in observed:
      assert current == f'execute_callback {name}'
    for s in callbacks:
      assert not s.attributes, s.attributes
      assert s.parent is not None and s.parent.span_id in parents
      assert s.end_time is not None
  return {
      'case': label,
      'callbacks_executed': len(observed),
      'callback_spans': len(callbacks),
      'callback_parents': {
          s.name: parents[s.parent.span_id] for s in callbacks
      },
      'final_output': 'done',
  }


async def main():
  results = [
      await one_case(*case)
      for case in [
          ('default_off', None, None),
          ('env_on', 'true', None),
          ('request_on', 'false', True),
          ('request_off', 'true', False),
      ]
  ]
  output = {'loaded_module': str(loaded), 'cases': results}
  args.result.write_text(json.dumps(output, indent=2), encoding='utf-8')
  print(json.dumps(output, indent=2))


asyncio.run(main())
