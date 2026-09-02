class AgentError(RuntimeError):
    """Base class for agent-level failures."""


class OutputValidationError(AgentError):
    """Raised when the LLM's response, after all retries, still doesn't
    parse as JSON or doesn't satisfy the IncidentOutput schema."""
