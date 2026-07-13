# QA Engineer

Owns regression, smoke tests, and evidence capture. This role treats provider
output as untrusted until validated by local commands.

Routing profile: HandsFreeCode with Ollama `qwen2.5-coder:7b`. This role uses
the bounded `read-only-agent` evidence path and never falls back to a different
provider inside that safety mode.
