"""Transport-only provider clients — one module per LLM provider.

Every client in this package implements agent.providers.base.BaseProviderClient
and nothing else: no orchestration, no retries beyond the single HTTP
call, no memory, no decision-making. ModelRouter (agent/model_router.py)
owns all of that.
"""
