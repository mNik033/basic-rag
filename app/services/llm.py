import json
from typing import AsyncGenerator, Optional
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
        keep_alive: Optional[str] = None,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
        temperature: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        think: Optional[bool] = None,
        num_thread: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout_seconds or settings.ollama_timeout_seconds
        self.keep_alive = keep_alive or settings.ollama_keep_alive
        self.num_predict = num_predict if num_predict is not None else settings.ollama_num_predict
        self.num_ctx = num_ctx if num_ctx is not None else settings.ollama_num_ctx
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else settings.ollama_repeat_penalty
        self.repeat_last_n = repeat_last_n if repeat_last_n is not None else settings.ollama_repeat_last_n
        self.top_k = top_k if top_k is not None else settings.ollama_top_k
        self.top_p = top_p if top_p is not None else settings.ollama_top_p
        self.think = think if think is not None else settings.ollama_think
        self.num_thread = num_thread if num_thread is not None else settings.ollama_num_thread

    def _build_options(self) -> dict:
        """Construct generation options dictionary for Ollama runtime."""
        opts = {
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        if self.num_thread is not None:
            opts["num_thread"] = self.num_thread
        return opts

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_json: bool = False,
    ) -> str:
        """Send a prompt to Ollama's /api/generate endpoint and return the completed response."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self._build_options(),
            "think": self.think,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_json:
            payload["format"] = "json"

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

    async def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response tokens asynchronously from Ollama's /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": self._build_options(),
            "think": self.think,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMResponseError(
                            status_code=response.status_code,
                            message=f"Ollama streaming returned HTTP error {response.status_code}: {body.decode(errors='replace')}",
                        )

                    async for line in response.aiter_lines():
                        if not line or not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise LLMConnectionError(
                    base_url=self.base_url,
                    error=f"Streaming connection to Ollama failed: {str(exc)}",
                ) from exc
            except httpx.RequestError as exc:
                raise LLMConnectionError(
                    base_url=self.base_url,
                    error=f"Streaming request to Ollama failed: {str(exc)}",
                ) from exc

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
