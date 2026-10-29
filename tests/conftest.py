import asyncio
import json
import re
from difflib import unified_diff

import pytest


class LoopStallProbe:
    """Heartbeat on the running event loop.

    Ticks every 10 ms while the ``async with`` body runs and records the
    longest time the loop went without giving it a turn.  Anything
    synchronous and slow that the body runs on the loop thread shows up as a
    stall of about its own duration; work that is awaited (I/O, a worker
    thread) does not.
    """

    TICK = 0.01

    def __init__(self):
        self.max_stall = 0.0
        self._stop = False
        self._task = None

    @property
    def max_stall_ms(self) -> float:
        return self.max_stall * 1000

    async def _beat(self):
        loop = asyncio.get_running_loop()
        last = loop.time()
        while not self._stop:
            await asyncio.sleep(self.TICK)
            now = loop.time()
            self.max_stall = max(self.max_stall, now - last - self.TICK)
            last = now

    async def __aenter__(self):
        self._task = asyncio.create_task(self._beat())
        # Let the heartbeat settle so its own start-up is not measured.
        await asyncio.sleep(self.TICK * 3)
        self.max_stall = 0.0
        return self

    async def __aexit__(self, *exc):
        # A stall that ends exactly when the body returns is only visible on
        # the next tick: give the heartbeat that tick before stopping it.
        await asyncio.sleep(self.TICK * 3)
        self._stop = True
        await self._task

    def assert_responsive(self, what: str, limit_ms: float = 100):
        # Shown for failures, and for passes with `pytest -rP`.
        print(f"[loop stall] {what}: {self.max_stall_ms:.0f} ms")
        assert self.max_stall_ms < limit_ms, (
            f"event loop stalled {self.max_stall_ms:.0f} ms during {what} "
            f"(limit {limit_ms:.0f} ms)"
        )


@pytest.fixture
def loop_probe():
    """A fresh event-loop heartbeat: ``async with loop_probe as probe: ...``."""
    return LoopStallProbe()


def replace_file_paths_with_XXX(content: str) -> str:
    """Replace file paths with XXX for consistent diffs across environments.

    Also normalizes traceback content to handle code refactoring where the
    stack trace structure may change but the error type/message stays the same.
    """
    if "Traceback" in content:
        # Replace entire traceback body with placeholder, keeping only the error line
        # This handles refactoring where stack frames change but the error is the same
        content = re.sub(
            r"Traceback \(most recent call last\):\\n(.*?)((?:TypeError|ValueError|AttributeError|KeyError|RuntimeError|Exception)[^\n]*)",
            r"Traceback (most recent call last):\\n  <stack frames>\\n\2",
            content,
            flags=re.DOTALL,
        )
    return content


def request_body_diff_matcher(r1, r2):
    """Matcher that compares requests with the same sequence index"""
    # Decode and pretty-format the bodies
    body1 = json.loads(r1.body.decode("utf-8"))
    body1_dump = json.dumps(body1, indent=2)
    body2 = json.loads(r2.body.decode("utf-8"))
    body2_dump = json.dumps(body2, indent=2)

    # body1.messages length != body2.messages length; we should not compare them
    if len(body1.get("messages", [])) != len(body2.get("messages", [])):
        return False

    body1_dump = replace_file_paths_with_XXX(body1_dump)
    body2_dump = replace_file_paths_with_XXX(body2_dump)

    # If bodies are different, show the diff
    if body1_dump != body2_dump:
        diff = unified_diff(
            body2_dump.splitlines(keepends=True),
            body1_dump.splitlines(keepends=True),
            n=3,
        )
        diff_as_string = (
            "\nRequest bodies differ! (- cassette, + current):\n" + "".join(diff)
        )
        pytest.fail(diff_as_string, pytrace=False)
    return True


def before_record_response(response):
    # Remove headers we don't want to record
    headers_to_filter = {
        "date",
        "server",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "anthropic-ratelimit-input-tokens-limit",
        "anthropic-ratelimit-input-tokens-remaining",
        "anthropic-ratelimit-input-tokens-reset",
        "anthropic-ratelimit-output-tokens-limit",
        "anthropic-ratelimit-output-tokens-remaining",
        "anthropic-ratelimit-output-tokens-reset",
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
        "anthropic-organization-id",
        "request-id",
        "openai-organization",
        "openai-processing-ms",
        "openai-version",
        "set-cookie",
        "cf-cache-status",
        "cf-ray",
    }

    response["headers"] = {
        k: v
        for k, v in response["headers"].items()
        if k.lower() not in headers_to_filter
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("Authorization", None),
            ("api-key", None),
            ("x-api-key", None),
            ("accept", None),
            ("accept-encoding", None),
            ("connection", None),
            ("content-length", None),
            ("user-agent", None),
            ("x-stainless-arch", None),
            ("x-stainless-async", None),
            ("x-stainless-lang", None),
            ("x-stainless-os", None),
            ("x-stainless-package-version", None),
            ("x-stainless-retry-count", None),
            ("x-stainless-runtime", None),
            ("x-stainless-runtime-version", None),
            ("cookie", None),
        ],
        "before_record_response": before_record_response,
        "record_mode": "none",
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "custom_body_diff",
        ],
    }


def pytest_recording_configure(config, vcr):
    vcr.register_matcher("custom_body_diff", request_body_diff_matcher)
