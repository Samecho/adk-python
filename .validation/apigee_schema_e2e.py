from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from google.adk.models import apigee_llm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import httpx
from pydantic import BaseModel


class Weather(BaseModel):
  city: str
  temperature_c: int


async def main():
  real_client = httpx.AsyncClient
  clients = []
  weather_schema = Weather.model_json_schema()
  wrapped_weather = {'name': 'Weather', 'schema': weather_schema}
  schema = types.Schema(
      type=types.Type.OBJECT,
      properties={'city': types.Schema(type=types.Type.STRING)},
  )
  raw_schema = {'type': 'object', 'properties': {'city': {'type': 'string'}}}
  cases = [
      ('output_schema', None, {'name': 'Weather', 'schema': weather_schema}),
      ('response_json_schema', weather_schema, wrapped_weather),
      ('types_schema', schema, {'name': 'response', 'schema': raw_schema}),
      ('wrapped_schema', wrapped_weather, wrapped_weather),
      ('json_only', None, None),
      ('no_schema', None, None),
  ]
  try:
    for case, value, expected in cases:
      for stream in (False, True):
        requests = []

        def respond(request):
          body = json.loads(request.content)
          requests.append(body)
          assert request.url.path == '/chat/completions'
          assert body['model'] == 'gpt-4o'
          if case == 'json_only':
            assert body['response_format'] == {'type': 'json_object'}
          elif case == 'no_schema':
            assert 'response_format' not in body
          else:
            assert body['response_format'] == {
                'type': 'json_schema',
                'json_schema': expected,
            }
          if stream:
            chunks = [
                {'choices': [{'index': 0, 'delta': {'content': '{}'}}]},
                {
                    'choices': [
                        {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
                    ]
                },
            ]
            data = ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks)
            return httpx.Response(
                200,
                headers={'Content-Type': 'text/event-stream'},
                content=(data + 'data: [DONE]\n\n').encode(),
            )
          return httpx.Response(
              200,
              json={
                  'choices': [{
                      'message': {'role': 'assistant', 'content': '{}'},
                      'finish_reason': 'stop',
                  }]
              },
          )

        def make_client(*args, **kwargs):
          kwargs['transport'] = httpx.MockTransport(respond)
          client = real_client(*args, **kwargs)
          clients.append(client)
          return client

        llm = apigee_llm.ApigeeLlm(
            model='apigee/openai/gpt-4o', proxy_url='https://proxy.example.test'
        )
        request = LlmRequest(
            model=llm.model,
            config=types.GenerateContentConfig(),
            contents=[
                types.Content(role='user', parts=[types.Part(text='Weather?')])
            ],
        )
        if case == 'output_schema':
          request.set_output_schema(Weather)
        elif case in ('response_json_schema', 'wrapped_schema'):
          request.config.response_json_schema = value
        elif case == 'types_schema':
          request.set_output_schema(value)
        elif case == 'json_only':
          request.config.response_mime_type = 'application/json'
        with patch.object(apigee_llm.httpx, 'AsyncClient', make_client):
          responses = [
              r
              async for r in llm.generate_content_async(request, stream=stream)
          ]
        assert len(requests) == 1
        assert any(
            part.text == '{}'
            for response in responses
            if response.content
            for part in response.content.parts or []
        )
        print(json.dumps({'case': case, 'stream': stream, 'result': 'passed'}))
  finally:
    for client in clients:
      await client.aclose()
  print('12 simulated end-to-end requests passed')


if __name__ == '__main__':
  print(f'Source: {apigee_llm.__file__}')
  asyncio.run(main())
