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

import warnings

from google.adk.utils.env_utils import is_enterprise_mode_enabled
from google.adk.utils.env_utils import is_env_enabled
import pytest


@pytest.mark.parametrize(
    'env_value,expected',
    [
        ('true', True),
        ('TRUE', True),
        ('TrUe', True),
        ('1', True),
        ('false', False),
        ('FALSE', False),
        ('0', False),
        ('', False),
    ],
)
def test_is_env_enabled(monkeypatch, env_value, expected):
  """Test is_env_enabled with various environment variable values."""
  monkeypatch.setenv('TEST_FLAG', env_value)
  assert is_env_enabled('TEST_FLAG') is expected


@pytest.mark.parametrize(
    'default,expected',
    [
        ('0', False),
        ('1', True),
        ('true', True),
    ],
)
def test_is_env_enabled_with_defaults(monkeypatch, default, expected):
  """Test is_env_enabled when env var is not set with different defaults."""
  monkeypatch.delenv('TEST_FLAG', raising=False)
  assert is_env_enabled('TEST_FLAG', default=default) is expected


def test_is_enterprise_mode_enabled_via_enterprise_env(monkeypatch):
  """Enterprise mode is on when GOOGLE_GENAI_USE_ENTERPRISE is truthy."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', 'true')

  assert is_enterprise_mode_enabled() is True


@pytest.mark.parametrize(
    'value,expected',
    [
        ('true', True),
        ('TRUE', True),
        ('1', True),
        ('false', False),
        ('0', False),
        ('', False),
    ],
)
def test_is_enterprise_mode_enabled_accepts_vertexai_alias_without_warning(
    monkeypatch, value, expected
):
  """The compatibility alias selects the backend without a runtime warning."""
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', value)

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    assert is_enterprise_mode_enabled() is expected


@pytest.mark.parametrize(
    'enterprise,expected', [('true', True), ('false', False), ('', False)]
)
@pytest.mark.parametrize('vertexai', ['true', 'false', ''])
def test_enterprise_env_takes_precedence_over_vertexai_alias(
    monkeypatch, enterprise, vertexai, expected
):
  """An explicitly set preferred variable wins, even when false or empty."""
  monkeypatch.setenv('GOOGLE_GENAI_USE_ENTERPRISE', enterprise)
  monkeypatch.setenv('GOOGLE_GENAI_USE_VERTEXAI', vertexai)

  with warnings.catch_warnings():
    warnings.simplefilter('error')
    assert is_enterprise_mode_enabled() is expected


def test_is_enterprise_mode_enabled_defaults_to_false(monkeypatch):
  """Enterprise mode is off when no relevant env var is set."""
  monkeypatch.delenv('GOOGLE_GENAI_USE_ENTERPRISE', raising=False)
  monkeypatch.delenv('GOOGLE_GENAI_USE_VERTEXAI', raising=False)

  assert is_enterprise_mode_enabled() is False
