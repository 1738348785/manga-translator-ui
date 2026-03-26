from .gemini import GeminiTranslator
from .keys import VERTEX_API_BASE


class VertexTranslator(GeminiTranslator):
    """Vertex text translator backed by the Gemini-compatible transport."""

    API_KEY_ENV = "VERTEX_API_KEY"
    API_BASE_ENV = None
    MODEL_ENV = "VERTEX_MODEL"
    DEFAULT_BASE_URL = VERTEX_API_BASE
    DEFAULT_MODEL_NAME = "gemini-1.5-flash"
    _GLOBAL_LAST_REQUEST_TS = {}

    def parse_args(self, args):
        super().parse_args(args)
        if self.base_url != self.DEFAULT_BASE_URL:
            self.base_url = self.DEFAULT_BASE_URL
            self.client = None
            self._setup_client()
