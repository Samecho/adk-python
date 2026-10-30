from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from google.adk.models import apigee_llm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import httpx


async def main() -> None:
  original_client = httpx.AsyncClient
  clients: list[httpx.AsyncClient] = []
  results = []
  cases = [
      ('gpt-5-mini', 'max_completion_tokens', 1024),
      ('o3', 'max_completion_tokens', 1024),
      ('o4-mini', 'max_completion_tokens', 1024),
      ('gpt-4o', 'max_tokens', 1024),
      ('gpt-5-chat-latest', 'max_tokens', 1024),
      ('qwen3', 'max_tokens', 1024),
      ('gpt-5-mini', 'max_completion_tokens', None),
      ('gpt-4o', 'max_tokens', None),
      ('gpt-5-mini', 'max_completion_tokens', 0),
      ('gpt-4o', 'max_tokens', 0),
  ]
  try:
    for model, expected_key, limit in cases:
      for stream in (False, True):
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
          body = json.loads(request.content)
          requests.append(body)
          assert request.url.path == '/chat/completions'
          assert body['model'] == model
          token_fields = {
              key: body[key]
              for key in ('max_tokens', 'max_completion_tokens')
              if key in body
          }
          expected = {} if limit is None else {expected_key: limit}
          if token_fields != expected:
            return httpx.Response(
                400,
                json={'error': {'message': f'{token_fields} != {expected}'}},
            )
          if stream:
            chunks = [
                {'choices': [{'index': 0, 'delta': {'content': 'OK'}}]},
                {
                    'choices': [
                        {'index': 0, 'delta': {}, 'finish_reason': 'stop'}
                    ]
                },
            ]
            data = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks)
            return httpx.Response(
                200,
                headers={'Content-Type': 'text/event-stream'},
                content=(data + 'data: [DONE]\n\n').encode(),
            )
          return httpx.Response(
              200,
              json={
                  'choices': [{
                      'message': {'role': 'assistant', 'content': 'OK'},
                      'finish_reason': 'stop',
                  }]
              },
          )

        def make_client(*args, **kwargs) -> httpx.AsyncClient:
          kwargs['transport'] = httpx.MockTransport(respond)
          client = original_client(*args, **kwargs)
          clients.append(client)
          return client

        llm = apigee_llm.ApigeeLlm(
            model=f'apigee/openai/{model}',
            proxy_url='https://proxy.example.test',
        )
        request = LlmRequest(
            model=llm.model,
            config=types.GenerateContentConfig(max_output_tokens=limit),
            contents=[
                types.Content(role='user', parts=[types.Part(text='Hi')])
            ],
        )
        try:
          with patch.object(apigee_llm.httpx, 'AsyncClient', make_client):
            responses = [
                r
                async for r in llm.generate_content_async(
                    request, stream=stream
                )
            ]
          assert len(requests) == 1
          assert any(
              p.text == 'OK'
              for r in responses
              if r.content
              for p in r.content.parts or []
          )
          result = 'passed'
        except httpx.HTTPStatusError as error:
          result = str(error.response.json()['error']['message'])
        entry = {
            'model': model,
            'stream': stream,
            'limit': limit,
            'result': result,
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)
  finally:
    for client in clients:
      await client.aclose()
  failures = [r for r in results if r['result'] != 'passed']
  print(
      json.dumps({'total': len(results), 'failed': len(failures)}), flush=True
  )
  assert not failures, failures


if __name__ == '__main__':
  print(f'SOURCE: {apigee_llm.__file__}', flush=True)
  asyncio.run(main())
