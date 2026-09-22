"""Where OllamaDecider sends requests. Precedence is explicit host > OLLAMA_HOST
env var > localhost:11434 -- the env var is the one the `ollama` CLI itself
reads, so a machine already pointed at a remote GPU box needs no extra config.
skip_capability_check=True so no network is touched.
"""

from __future__ import annotations

from agent.deciders import DEFAULT_OLLAMA_HOST, OllamaDecider, default_decider, ollama_host_default


def _decider(**kw) -> OllamaDecider:
    return OllamaDecider(skip_capability_check=True, **kw)


def test_defaults_to_localhost_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_host_default() == DEFAULT_OLLAMA_HOST
    assert _decider().host == "http://localhost:11434"


def test_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434/")
    assert _decider().host == "http://gpu-box:11434"  # trailing slash stripped


def test_bare_host_port_in_env_gets_a_scheme(monkeypatch):
    # The ollama CLI accepts OLLAMA_HOST=host:port with no scheme; so do we.
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    assert _decider().host == "http://gpu-box:11434"


def test_explicit_host_beats_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert _decider(host="http://other:9999").host == "http://other:9999"


def test_default_decider_threads_host_through_on_the_auto_path(monkeypatch):
    # Regression: --backend auto used to ignore --ollama-host entirely.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(OllamaDecider, "_model_capabilities", lambda self: None)
    d = default_decider(ollama_host="http://gpu-box:11434")
    assert isinstance(d, OllamaDecider)
    assert d.host == "http://gpu-box:11434"
