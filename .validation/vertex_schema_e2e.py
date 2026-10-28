import argparse
import asyncio
import datetime
import json
import os

from google import genai
from google.adk.agents.llm_agent import LlmAgent
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from google.oauth2.credentials import Credentials
import httpx
import pydantic


class Address(pydantic.BaseModel):
  city: str


class Person(pydantic.BaseModel):
  name: str
  address: Address


def schedule(person: Person, when: datetime.datetime) -> str:
  """Schedule a visit for a person at the given time."""
  return 'scheduled'


async def run(expect_legacy_error):
  os.environ['GOOGLE_GENAI_USE_ENTERPRISE'] = '1'
  results = []
  for json_schema_enabled in (True, False):
    payloads = []

    def respond(request):
      payload = json.loads(request.content)
      declaration = payload['tools'][0]['functionDeclarations'][0]
      if not json_schema_enabled:
        parameters = declaration['parameters']
        person = parameters['properties']['person']
        assert person['properties']['address']['properties']['city']['type'] == 'STRING'
        assert parameters['properties']['when']['format'] == 'date-time'
        assert '$ref' not in json.dumps(parameters)
        assert '$defs' not in json.dumps(parameters)
      payloads.append(payload)
      return httpx.Response(200, json={
          'candidates': [{
              'content': {'role': 'model', 'parts': [{'text': 'Ready to schedule.'}]},
              'finishReason': 'STOP',
          }],
          'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 3},
          'modelVersion': 'gemini-test',
      })

    transport = httpx.MockTransport(respond)
    sync_client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)
    client = genai.Client(
        enterprise=True,
        project='test-project',
        location='us-central1',
        credentials=Credentials(token='test-token'),
        http_options=types.HttpOptions(
            api_version='v1',
            httpx_client=sync_client,
            httpx_async_client=async_client,
        ),
    )
    runner = InMemoryRunner(agent=LlmAgent(
        name='scheduler',
        model=Gemini(model='gemini-test', client=client),
        tools=[FunctionTool(schedule)],
    ))
    try:
      session = await runner.session_service.create_session(
          app_name=runner.app_name, user_id='test-user'
      )
      with temporary_feature_override(
          FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema_enabled
      ):
        try:
          events = [event async for event in runner.run_async(
              user_id='test-user',
              session_id=session.id,
              new_message=types.Content(
                  role='user', parts=[types.Part(text='Schedule a visit.')]
              ),
          )]
        except ValueError as error:
          if not expect_legacy_error or json_schema_enabled:
            raise
          assert 'Failed to parse the parameter person' in str(error)
          assert not payloads
          results.append({'json_schema_enabled': False, 'result': 'expected declaration error', 'requests': 0})
          continue
      assert json_schema_enabled or not expect_legacy_error
      assert len(payloads) == 1
      assert events[-1].content.parts[0].text == 'Ready to schedule.'
      results.append({'json_schema_enabled': json_schema_enabled, 'result': 'passed', 'requests': len(payloads)})
    finally:
      await runner.close()
      await client.aio.aclose()
      client.close()
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--expect-legacy-error', action='store_true')
  parser.add_argument('--expect-installed', action='store_true')
  args = parser.parse_args()
  if args.expect_installed:
    import google.adk
    assert 'site-packages' in google.adk.__file__, google.adk.__file__
    print('Using installed wheel:', google.adk.__file__)
  asyncio.run(run(args.expect_legacy_error))
