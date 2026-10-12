"""AtlasKit — build on N-ATLaS, Nigeria's multilingual LLM, in a few lines."""
from .client import LANGUAGES, AtlasKit, AtlasKitError, ChatResponse, Conversation

__all__ = ["AtlasKit", "AtlasKitError", "ChatResponse", "Conversation", "LANGUAGES"]
__version__ = "0.1.0"
