from typing import Optional
import httpx
from app.core.config import get_settings
from app.core.exceptions import LLMConnectionError, LLMResponseError


class OllamaLLMService:
    """Service client for generating text responses using a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout_seconds or settings.ollama_timeout_seconds

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a prompt to Ollama's /api/generate endpoint and return the completed response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise LLMConnectionError(
                    base_url=self.base_url,
                    error=f"Connection to Ollama failed: {str(exc)}",
                ) from exc
            except httpx.RequestError as exc:
                raise LLMConnectionError(
                    base_url=self.base_url,
                    error=f"Request to Ollama failed: {str(exc)}",
                ) from exc

            if response.status_code != 200:
                raise LLMResponseError(
                    status_code=response.status_code,
                    message=f"Ollama returned HTTP error {response.status_code}: {response.text}",
                )

            data = response.json()
            return data.get("response", "").strip()

    async def is_healthy(self) -> bool:
        """Check if Ollama server is reachable and active."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False


def get_llm_service() -> OllamaLLMService:
    """Dependency / accessor function for OllamaLLMService."""
    return OllamaLLMService()
