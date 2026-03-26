from .gemini_hq import GeminiHighQualityTranslator
from .keys import VERTEX_API_BASE


class VertexHighQualityTranslator(GeminiHighQualityTranslator):
    """Vertex high-quality translator backed by the Gemini-compatible transport."""

    API_KEY_ENV = "VERTEX_API_KEY"
    API_BASE_ENV = None
    MODEL_ENV = "VERTEX_MODEL"
    DEFAULT_BASE_URL = VERTEX_API_BASE
    DEFAULT_MODEL_NAME = "gemini-1.5-flash"
    LOG_PROVIDER_NAME = "Vertex HQ"
    LOG_PROVIDER_NAME_ZH = "Vertex高质量翻译"
    STREAM_LOG_PREFIX = "[Vertex HQ Stream]"
    _GLOBAL_LAST_REQUEST_TS = {}

    def parse_args(self, args):
        super().parse_args(args)
        if self.base_url != self.DEFAULT_BASE_URL:
            self.base_url = self.DEFAULT_BASE_URL
            self.client = None
            self._setup_client()
