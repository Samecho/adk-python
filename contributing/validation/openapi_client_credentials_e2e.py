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
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.parse import parse_qs

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
parser.add_argument('--result', required=True, type=Path)
options = parser.parse_args()
if options.source:
  sys.path.insert(0, str(options.source.resolve()))

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlows
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.auth import AuthCredential
from google.adk.auth import AuthCredentialTypes
from google.adk.auth import OAuth2Auth
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
import google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler as auth_module
from google.genai import types
import httpx
from requests import Response

loaded = Path(auth_module.__file__).resolve()
if options.source:
  assert loaded.is_relative_to(options.source.resolve())
else:
  assert 'site-packages' in loaded.parts
token_calls = []
api_calls = []


def token_send(request, **kwargs):
  assert request.url.startswith('https://token.example.test/')
  raw = (
      request.body.decode() if isinstance(request.body, bytes) else request.body
  )
  token_calls.append({'method': request.method, 'body': parse_qs(raw)})
  response = Response()
  response.status_code = 200
  response.headers['Content-Type'] = 'application/json'
  response._content = (
      b'{"access_token":"dummy-access","token_type":"Bearer","expires_in":3600}'
  )
  response.request = request
  return response


def api_send(request):
  api_calls.append({
      'method': request.method,
      'authorization': request.headers.get('authorization'),
  })
  return httpx.Response(200, json={'value': 2})


class Model(BaseLlm):

  async def generate_content_async(self, llm_request, stream=False):
    has_response = any(
        part.function_response
        for c in llm_request.contents
        for part in c.parts or []
    )
    part = (
        types.Part(text='done')
        if has_response
        else types.Part.from_function_call(name='lookup', args={})
    )
    yield LlmResponse(content=types.Content(role='model', parts=[part]))


async def main():
  operation = {
      'operationId': 'lookup',
      'responses': {
          '200': {
              'description': 'ok',
              'content': {'application/json': {'schema': {'type': 'object'}}},
          }
      },
  }
  spec = {
      'openapi': '3.0.0',
      'info': {'title': 'probe', 'version': '1'},
      'servers': [{'url': 'https://api.example.test'}],
      'paths': {'/lookup': {'get': operation}},
  }
  scheme = OAuth2(
      flows=OAuthFlows(
          clientCredentials=OAuthFlowClientCredentials(
              tokenUrl='https://token.example.test/oauth/token',
              scopes={'read': 'Read'},
          )
      )
  )
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(client_id='dummy-client', client_secret='dummy-secret'),
  )
  toolset = OpenAPIToolset(
      spec_dict=spec,
      auth_scheme=scheme,
      auth_credential=credential,
      httpx_client_factory=lambda: httpx.AsyncClient(
          transport=httpx.MockTransport(api_send)
      ),
  )
  agent = LlmAgent(name='probe', model=Model(model='scripted'), tools=[toolset])
  with (
      patch('requests.sessions.Session.send', side_effect=token_send),
      patch(
          'webbrowser.open', side_effect=AssertionError('Unexpected browser')
      ),
  ):
    async with Runner(
        app_name='probe_app',
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    ) as runner:
      events = [
          e
          async for e in runner.run_async(
              user_id='user',
              session_id='session',
              new_message=types.Content(
                  role='user', parts=[types.Part(text='lookup')]
              ),
              run_config=RunConfig(max_llm_calls=3),
          )
      ]
  requests = []
  for event in events:
    for _, config in (event.actions.requested_auth_configs or {}).items():
      requests.append({
          'auth_uri': (
              config.exchanged_auth_credential.oauth2.auth_uri
              if config.exchanged_auth_credential
              and config.exchanged_auth_credential.oauth2
              else None
          )
      })
  result = {
      'validated_commit': '8d9f57d2af8641da78d5b1689739676f35da3f69',
      'loaded_module': str(loaded),
      'module_sha256': hashlib.sha256(loaded.read_text().encode()).hexdigest(),
      'token_calls': token_calls,
      'api_calls': api_calls,
      'requested_auth_configs': requests,
      'function_calls': [
          c.name for e in events for c in e.get_function_calls()
      ],
      'function_responses': [
          {'name': r.name, 'response': r.response}
          for e in events
          for r in e.get_function_responses()
      ],
  }
  assert len(token_calls) == 1 and len(api_calls) == 1
  assert not requests
  assert token_calls[0]['body']['grant_type'] == ['client_credentials']
  assert api_calls[0]['authorization'] == 'Bearer dummy-access'
  options.result.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(json.dumps(result, indent=2))


asyncio.run(main())
