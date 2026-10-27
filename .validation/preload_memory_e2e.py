from collections.abc import AsyncGenerator
import json
from pathlib import Path
import sys

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types
from pydantic import Field


class ProbeLlm(BaseLlm):
  model: str = 'scripted-memory-probe'
  requests: list[LlmRequest] = Field(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request.model_copy(deep=True))
    yield LlmResponse(content=types.ModelContent('ready'))


async def main() -> None:
  import google.adk.tools.preload_memory_tool as module

  module_path = Path(module.__file__).resolve()
  if '--expect-installed' in sys.argv:
    assert 'site-packages' in module_path.parts, str(module_path)

  sessions = InMemorySessionService()
  memory = InMemoryMemoryService()
  past = await sessions.create_session(
      app_name='memory_probe', user_id='user', session_id='past'
  )
  await sessions.append_event(
      session=past,
      event=Event(
          author='user',
          invocation_id='seed',
          content=types.UserContent('My favorite tea is green tea.'),
      ),
  )
  saved = await sessions.get_session(
      app_name='memory_probe', user_id='user', session_id='past'
  )
  assert saved is not None
  await memory.add_session_to_memory(saved)

  cases = [
      {'instruction': 'Use these remembered preferences when answering.'},
      {'instruction': ''},
      {'instruction': 'Remembered topics: {preferences}\nUse relevant details.'},
      {},
  ]
  blocks, system_instructions, results = [], [], []
  for index, kwargs in enumerate(cases):
    model = ProbeLlm()
    agent = LlmAgent(
        name='memory_agent',
        model=model,
        instruction='Keep this system instruction unchanged.',
        tools=[PreloadMemoryTool(**kwargs)],
    )
    session_id = f'current-{index}'
    await sessions.create_session(
        app_name='memory_probe', user_id='user', session_id=session_id
    )
    async with Runner(
        app=App(name='memory_probe', root_agent=agent),
        session_service=sessions,
        memory_service=memory,
    ) as runner:
      events = [event async for event in runner.run_async(
          user_id='user',
          session_id=session_id,
          new_message=types.UserContent('What tea do I like?'),
      )]
    assert any(
        part.text == 'ready'
        for event in events
        for part in (event.content.parts if event.content else [])
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    recalled = [
        part.text
        for content in request.contents
        for part in content.parts or []
        if part.text and '<PAST_CONVERSATIONS>' in part.text
    ]
    assert len(recalled) == 1
    text = recalled[0]
    assert 'My favorite tea is green tea.' in text
    if 'instruction' in kwargs:
      expected_prefix = kwargs['instruction']
      assert text.split('<PAST_CONVERSATIONS>')[0] == (
          expected_prefix + '\n' if expected_prefix else ''
      )
    else:
      assert text.startswith(
          'The following content is from your previous conversations with the user.\n'
          "They may be useful for answering the user's current query.\n"
      )
    blocks.append(text[text.index('<PAST_CONVERSATIONS>'):])
    system_instructions.append(request.config.system_instruction)
    results.append({'case': index, 'configured': bool(kwargs), 'passed': True})

  assert all(block == blocks[0] for block in blocks)
  assert all(value == system_instructions[0] for value in system_instructions)
  print(json.dumps({'module': str(module_path), 'cases': results,
                    'memory_block_unchanged': True, 'system_prefix_unchanged': True,
                    'live_model_calls': 0}))


if __name__ == '__main__':
  import asyncio

  asyncio.run(main())
