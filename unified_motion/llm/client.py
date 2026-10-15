"""Small, bounded DeepSeek/GLM Chat Completions adapter with injectable transport."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


PROVIDERS = {
    'deepseek': ('https://api.deepseek.com/chat/completions', 'DEEPSEEK_API_KEY'),
    'glm': ('https://open.bigmodel.cn/api/paas/v4/chat/completions', 'GLM_API_KEY'),
}


class AdvisorAPIError(RuntimeError):
    """Sanitized provider error; never include headers, credentials or response bodies."""


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    model: str
    timeout_seconds: float = 30.
    max_tokens: int = 2048
    retries: int = 1
    json_mode: bool = True
    thinking: bool | None = None

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError('provider must be deepseek or glm')
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 120:
            raise ValueError('specify a model available to your provider account')
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise ValueError('timeout_seconds must be in (0,120]')
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192:
            raise ValueError('max_tokens must be in [1,8192]')
        if type(self.retries) is not int or not 0 <= self.retries <= 2:
            raise ValueError('retries must be in [0,2]')
        if type(self.json_mode) is not bool or (self.thinking is not None and type(self.thinking) is not bool):
            raise ValueError('invalid response mode')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a provider key to a redirect target.


def _transport(url, payload, key, timeout):
    request = Request(url, data=json.dumps(payload, allow_nan=False).encode('utf-8'),
        headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise AdvisorAPIError('provider response exceeds 1 MiB')
    return json.loads(raw.decode('utf-8'))


class ChatClient:
    def __init__(self, config, *, transport=None, sleep=time.sleep):
        self.config = config
        self.transport = transport or _transport
        self.sleep = sleep

    def complete(self, system, user):
        cfg = self.config
        if (not isinstance(system, str) or not isinstance(user, str)
                or len((system+user).encode('utf-8')) > 131072):
            raise ValueError('advisor prompt must contain at most 128 KiB of text')
        endpoint, variable = PROVIDERS[cfg.provider]
        key = os.environ.get(variable)
        if not key:
            raise AdvisorAPIError(f'set {variable} in the training process environment')
        payload = dict(model=cfg.model, messages=[{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}], stream=False, max_tokens=cfg.max_tokens,
            temperature=.2)
        if cfg.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        if cfg.thinking is not None:
            payload['thinking'] = {'type': 'enabled' if cfg.thinking else 'disabled'}
        started = time.perf_counter()
        for attempt in range(cfg.retries+1):
            try:
                result = self.transport(endpoint, payload, key, cfg.timeout_seconds)
                break
            except HTTPError as error:
                if attempt < cfg.retries and (error.code == 429 or 500 <= error.code < 600):
                    self.sleep(min(2**attempt, 4))
                    continue
                raise AdvisorAPIError(f'{cfg.provider} HTTP {error.code}') from None
            except (URLError, TimeoutError, OSError):
                # A timed-out request may already have incurred a charge; do not retry it.
                raise AdvisorAPIError(f'{cfg.provider} connection or timeout failure') from None
            except (ValueError, UnicodeError):
                raise AdvisorAPIError('provider returned invalid JSON') from None
        try:
            choice = result['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise AdvisorAPIError('provider response incomplete or not a text answer')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip() or len(content) > 32768:
                raise AdvisorAPIError('provider returned empty or oversized content')
            usage = {k: v for k, v in result.get('usage', {}).items()
                     if k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                     and type(v) is int and v >= 0}
        except (KeyError, IndexError, TypeError, AttributeError):
            raise AdvisorAPIError('provider response does not match Chat Completions') from None
        return content, dict(provider=cfg.provider, model=cfg.model, usage=usage,
                             api_seconds=time.perf_counter()-started, attempts=attempt+1)
