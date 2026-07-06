"""All tunables live under config/ — timeouts, quotas, model names,
logging. Nothing elsewhere in the codebase should read os.environ
directly or hardcode a rule-governed constant (REG-10)."""
