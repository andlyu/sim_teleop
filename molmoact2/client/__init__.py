"""Mac-side MolmoAct2 client + policy seam (no GPU deps)."""
from .policy import Observation, Policy, StubPolicy

__all__ = ["Observation", "Policy", "StubPolicy", "MolmoActClient"]


def __getattr__(name):
    # Lazy: importing the client pulls in `requests`, keep it optional.
    if name == "MolmoActClient":
        from .molmoact_client import MolmoActClient

        return MolmoActClient
    raise AttributeError(name)
