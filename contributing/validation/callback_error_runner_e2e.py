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

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.utils import _callback_pipeline
from google.genai import errors as genai_errors
from google.genai import types

loaded = Path(_callback_pipeline.__file__).resolve()
if args.source:
  assert loaded.is_relative_to(args.source.resolve()), loaded
else:
  assert 'site-packages' in loaded.parts, loaded

model_calls = []


class ScriptedModel(BaseLlm):

  async def generate_content_async(self, llm_request, stream=False):
    model_calls.append(True)
    yield LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text='done')])
    )


async def one_case(schema, enabled, is_async, error, expected_type):
  exporter.clear()
  model_calls.clear()
  os.environ['ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN'] = schema
  os.environ['ADK_EXPERIMENTAL_TELEMETRY'] = str(enabled).lower()

  def callback(callback_context):
    raise error

  async def async_callback(callback_context):
    raise error

  agent = LlmAgent(
      name='callback_probe',
      model=ScriptedModel(model='scripted'),
      before_agent_callback=async_callback if is_async else callback,
  )
  async with Runner(
      app_name='callback_probe',
      agent=agent,
      session_service=InMemorySessionService(),
      auto_create_session=True,
  ) as runner:
    try:
      async for _ in runner.run_async(
          user_id='user',
          session_id='session',
          new_message=types.Content(
              role='user', parts=[types.Part(text='hello')]
          ),
      ):
        pass
    except type(error) as caught:
      assert caught is error
    else:
      raise AssertionError('Callback error did not reach the caller')

  assert not model_calls
  provider.force_flush()
  spans = [
      span for span in exporter.get_finished_spans()
      if span.name.startswith('execute_callback ')
  ]
  assert len(spans) == int(enabled)
  if enabled:
    (span,) = spans
    assert span.status.status_code.name == 'ERROR'
    assert span.attributes['error.type'] == expected_type
    assert span.events[0].name == 'exception'
  return {
      'schema': schema,
      'telemetry_enabled': enabled,
      'callback': 'async' if is_async else 'sync',
      'expected_error_type': expected_type,
      'callback_spans': len(spans),
      'attributes': dict(spans[0].attributes) if enabled else {},
      'original_error_propagated': True,
      'model_called': False,
  }


async def main():
  cases = []
  for schema in ('1', '2'):
    for enabled in (False, True):
      for is_async in (False, True):
        for error, expected in [
            (ValueError('callback failed'), 'ValueError'),
            (genai_errors.ClientError(429, {'message': 'quota exceeded'}), '429'),
        ]:
          cases.append(await one_case(schema, enabled, is_async, error, expected))
  result = {'loaded_module': str(loaded), 'cases': cases}
  args.result.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
  provider.shutdown()
  print(json.dumps({'cases_passed': len(cases), 'loaded_module': str(loaded)}))


asyncio.run(main())
