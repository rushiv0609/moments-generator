"""
Pluggable LLM Interface for Director Agent state machine.
Provides Ollama integration with structured output schemas and mock fallback.
"""

from abc import ABC, abstractmethod
import json
import logging
import time
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel

from app.core.director.state import PlannerOutput, DraftingOutput, EditorOutput

logger = logging.getLogger(__name__)


class DirectorLLMInterface(ABC):
    """Abstract interface for Director Agent LLM backends."""

    @abstractmethod
    def structured_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
    ) -> BaseModel:
        """
        Generate structured output matching the provided Pydantic schema.
        """
        pass

    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return model metadata (name, context length, etc.)."""
        pass

    def unload(self) -> bool:
        """Unload model from GPU/RAM after execution."""
        return False


class OllamaDirectorLLM(DirectorLLMInterface):
    """
    Ollama backend for Apple Silicon local inference.
    Uses ChatOllama with structured output support.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        fallback_to_mock: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.fallback_to_mock = fallback_to_mock
        self._mock = MockDirectorLLM() if fallback_to_mock else None

        # Auto-ensure Ollama daemon is active
        try:
            from app.core.ollama_service import ensure_ollama_running
            self._available = ensure_ollama_running(base_url=base_url)
        except Exception:
            self._available = False

        # Try initializing ChatOllama
        try:
            from langchain_ollama import ChatOllama
            self._chat = ChatOllama(
                model=model_name,
                base_url=base_url,
            )
        except Exception as e:
            logger.warning("Failed to initialize ChatOllama (%s). Fallback enabled: %s", e, fallback_to_mock)
            self._available = False
            self._chat = None

        self.last_telemetry: Dict[str, Any] = {}

    def structured_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
    ) -> BaseModel:
        start_t = time.time()
        # Check / start Ollama before generation
        try:
            from app.core.ollama_service import ensure_ollama_running
            self._available = ensure_ollama_running(base_url=self.base_url)
        except Exception:
            pass

        if not self._available or self._chat is None:
            if self._mock:
                logger.info("Ollama unavailable, using mock generator for %s", response_schema.__name__)
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                return res
            raise RuntimeError(f"Ollama server is unavailable at {self.base_url}")

        try:
            import json
            import re
            from langchain_core.messages import SystemMessage, HumanMessage
            from langchain_ollama import ChatOllama

            # Build configured chat model with format="json" and num_predict option
            chat = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                format="json",
                options={
                    "num_predict": 1024,
                    "temperature": temperature,
                },
            )
            
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            enhanced_sys_prompt = (
                f"{system_prompt}\n\n"
                f"You MUST output valid JSON matching this exact JSON Schema:\n{schema_json}\n"
                f"Output raw JSON only. Do NOT output markdown explanations or preambles."
            )

            messages = [
                SystemMessage(content=enhanced_sys_prompt),
                HumanMessage(content=user_prompt),
            ]

            result_obj = None
            raw_text = ""

            try:
                structured_llm = chat.with_structured_output(response_schema)
                result = structured_llm.invoke(messages)
                if isinstance(result, response_schema):
                    result_obj = result
                elif isinstance(result, dict):
                    result_obj = response_schema.model_validate(result)
                elif result is not None:
                    result_obj = result
            except Exception as parse_err:
                logger.debug("Structured output helper parsing error (%s), attempting direct JSON block extraction...", parse_err)
                raw_resp = chat.invoke(messages)
                raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
                # Extract first matching JSON object block
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    parsed_dict = json.loads(match.group(0))
                    result_obj = response_schema.model_validate(parsed_dict)
                else:
                    raise parse_err

            elapsed = round(time.time() - start_t, 3)
            self.last_telemetry = {
                "backend": "ollama",
                "model": self.model_name,
                "latency_seconds": elapsed,
                "schema": response_schema.__name__,
                "prompt_preview": user_prompt[:180] + ("..." if len(user_prompt) > 180 else ""),
                "full_prompt": user_prompt,
                "system_prompt": system_prompt,
                "response_json": result_obj.model_dump() if result_obj else {},
            }
            logger.info("Ollama [%s] completed schema %s in %.2fs", self.model_name, response_schema.__name__, elapsed)

            return result_obj if result_obj is not None else response_schema.model_construct()
        except Exception as e:
            elapsed = round(time.time() - start_t, 3)
            logger.warning("Ollama call failed after %.2fs (%s). Checking fallback.", elapsed, e)
            if self._mock:
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                return res
            raise

    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "available": self._available,
        }

    def unload(self) -> bool:
        """
        Unload the local LLM model from GPU/VRAM immediately via Ollama keep_alive=0 API.
        Keeps the Ollama server daemon alive while freeing all GPU and RAM.
        """
        if not self._available or not self.model_name:
            return False
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info("Successfully unloaded Ollama model '%s' from GPU/VRAM (freed memory)", self.model_name)
                return True
        except Exception as e:
            logger.debug("Notice while unloading Ollama model '%s': %s", self.model_name, e)
        return False


class MockDirectorLLM(DirectorLLMInterface):
    """
    Deterministic Mock LLM for unit tests, offline operation, and benchmark simulation.
    """

    def __init__(self, model_name: str = "mock-director-llm"):
        self.model_name = model_name
        self.last_telemetry: Dict[str, Any] = {}

    def structured_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
    ) -> BaseModel:
        start_t = time.time()
        res = None
        if response_schema == PlannerOutput:
            res = PlannerOutput(
                search_queries=[
                    "scenic landscape mountain view",
                    "friends laughing candid moment",
                    "sunset golden hour outdoors",
                    "action movement activity",
                ],
                mood_or_narrative="A dynamic and emotional cinematic journey celebrating friendship and nature.",
                target_duration_seconds=30,
            )
        elif response_schema == DraftingOutput:
            res = DraftingOutput(
                storyboard=[],
                narrative_arc="Mock curated timeline spanning opening scenic shots to concluding golden hour moments.",
            )
        elif response_schema == EditorOutput:
            res = EditorOutput(
                approved=True,
                feedback="Storyboard meets target duration and has balanced scene diversity.",
                pacing_score=8.5,
                suggested_modifications=[],
            )
        else:
            res = response_schema.model_construct()

        elapsed = round(time.time() - start_t, 4)
        self.last_telemetry = {
            "backend": "mock",
            "model": self.model_name,
            "latency_seconds": elapsed,
            "schema": response_schema.__name__,
            "prompt_preview": user_prompt[:180],
            "full_prompt": user_prompt,
            "system_prompt": system_prompt,
            "response_json": res.model_dump(),
        }
        return res

    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "mock",
            "model_name": self.model_name,
            "available": True,
        }


class GeminiDirectorLLM(DirectorLLMInterface):
    """
    Google Gemini Cloud backend using high-speed REST API with native JSON Schema output.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.7-flash",
        api_key: Optional[str] = None,
        fallback_to_mock: bool = False,
    ):
        import os
        from app.config import get_settings
        settings = get_settings()

        self.model_name = model_name
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        self.fallback_to_mock = fallback_to_mock
        self._mock = MockDirectorLLM(model_name=f"mock-{model_name}") if fallback_to_mock else None
        self.last_telemetry: Dict[str, Any] = {}

    def structured_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
    ) -> BaseModel:
        start_t = time.time()

        if not self.api_key:
            if self._mock:
                logger.warning("No Gemini API key provided. Falling back to mock generator.")
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                self.last_telemetry["backend"] = "gemini (mock-fallback: missing key)"
                return res
            raise ValueError("GEMINI_API_KEY is not configured. Please provide a valid Gemini API key.")

        import httpx
        import json
        import re

        clean_model = self.model_name.replace("gemini/", "").replace("google/", "")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={self.api_key}"

        # Convert Pydantic schema to Gemini OpenAPI 3.0 schema with full $ref dereferencing
        schema_dict = response_schema.model_json_schema()
        defs_pool = schema_dict.get("$defs") or schema_dict.get("definitions") or {}

        def _resolve_gemini_schema(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_key = node["$ref"].split("/")[-1]
                    if ref_key in defs_pool:
                        resolved = dict(defs_pool[ref_key])
                        return _resolve_gemini_schema(resolved)
                cleaned = {}
                for k, v in node.items():
                    if k in ("title", "$defs", "definitions"):
                        continue
                    cleaned[k] = _resolve_gemini_schema(v)
                return cleaned
            elif isinstance(node, list):
                return [_resolve_gemini_schema(item) for item in node]
            return node

        gemini_schema = _resolve_gemini_schema(schema_dict)

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_schema": gemini_schema,
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()

            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(raw_text)
            result_obj = response_schema.model_validate(parsed_json)

            elapsed = round(time.time() - start_t, 3)
            self.last_telemetry = {
                "backend": "gemini",
                "model": self.model_name,
                "latency_seconds": elapsed,
                "schema": response_schema.__name__,
                "prompt_preview": user_prompt[:180] + ("..." if len(user_prompt) > 180 else ""),
                "full_prompt": user_prompt,
                "system_prompt": system_prompt,
                "response_json": result_obj.model_dump(),
            }
            logger.info("Gemini [%s] completed schema %s in %.2fs", self.model_name, response_schema.__name__, elapsed)
            return result_obj

        except Exception as e:
            elapsed = round(time.time() - start_t, 3)
            err_msg = str(e)
            logger.warning("Gemini call [%s] failed after %.2fs (%s). Checking fallback.", self.model_name, elapsed, err_msg)
            if self._mock:
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                self.last_telemetry["backend"] = f"gemini (mock-fallback: {type(e).__name__} - {err_msg[:60]})"
                return res
            raise

    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "gemini",
            "model_name": self.model_name,
            "available": bool(self.api_key),
        }


class GroqDirectorLLM(DirectorLLMInterface):
    """
    Groq Cloud backend using ultra-fast LPUs with OpenAI-compatible JSON mode.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        fallback_to_mock: bool = False,
    ):
        import os
        from app.config import get_settings
        settings = get_settings()

        self.model_name = model_name
        self.api_key = api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY") or ""
        self.fallback_to_mock = fallback_to_mock
        self._mock = MockDirectorLLM(model_name=f"mock-{model_name}") if fallback_to_mock else None
        self.last_telemetry: Dict[str, Any] = {}

    def structured_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
    ) -> BaseModel:
        start_t = time.time()

        if not self.api_key:
            if self._mock:
                logger.warning("No Groq API key provided. Falling back to mock generator.")
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                self.last_telemetry["backend"] = "groq (mock-fallback: missing key)"
                return res
            raise ValueError("GROQ_API_KEY is not configured. Please provide a valid Groq API key.")

        import httpx
        import json
        import re

        clean_model = self.model_name.replace("groq:", "").replace("groq/", "")
        endpoint = "https://api.groq.com/openai/v1/chat/completions"

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        enhanced_sys_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST output valid JSON conforming exactly to this JSON schema:\n{schema_json}\n"
            f"Do not include any conversational filler, markdown fences, or explanations. Output pure JSON only."
        )

        payload = {
            "model": clean_model,
            "messages": [
                {"role": "system", "content": enhanced_sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=25.0) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            raw_text = data["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_text)
            result_obj = response_schema.model_validate(parsed_json)

            elapsed = round(time.time() - start_t, 3)
            self.last_telemetry = {
                "backend": "groq",
                "model": self.model_name,
                "latency_seconds": elapsed,
                "schema": response_schema.__name__,
                "prompt_preview": user_prompt[:180] + ("..." if len(user_prompt) > 180 else ""),
                "full_prompt": user_prompt,
                "system_prompt": system_prompt,
                "response_json": result_obj.model_dump(),
            }
            logger.info("Groq [%s] completed schema %s in %.2fs", self.model_name, response_schema.__name__, elapsed)
            return result_obj

        except Exception as e:
            elapsed = round(time.time() - start_t, 3)
            logger.warning("Groq call failed after %.2fs (%s). Checking fallback.", elapsed, e)
            if self._mock:
                res = self._mock.structured_generate(system_prompt, user_prompt, response_schema, temperature)
                self.last_telemetry = self._mock.last_telemetry
                self.last_telemetry["backend"] = "groq (mock-fallback: error)"
                return res
            raise

    def model_info(self) -> Dict[str, Any]:
        return {
            "backend": "groq",
            "model_name": self.model_name,
            "available": bool(self.api_key),
        }


def get_director_llm(
    model_name: str = "gemma4:e4b-mlx",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    fallback_to_mock: bool = False,
) -> DirectorLLMInterface:
    """
    Factory resolving Director LLM backend based on model_name prefix / provider.
    """
    m_lower = model_name.lower()

    # 1. Gemini Cloud Models
    if m_lower.startswith("gemini") or "gemini-" in m_lower:
        return GeminiDirectorLLM(
            model_name=model_name,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    # 2. Groq Cloud Models
    if m_lower.startswith("groq") or "llama-3" in m_lower or "mixtral" in m_lower:
        return GroqDirectorLLM(
            model_name=model_name,
            api_key=api_key,
            fallback_to_mock=fallback_to_mock,
        )

    # 3. Deterministic Mock
    if m_lower.startswith("mock"):
        return MockDirectorLLM(model_name=model_name)

    # 4. Default Local Ollama
    return OllamaDirectorLLM(
        model_name=model_name,
        base_url=base_url or "http://localhost:11434",
        fallback_to_mock=fallback_to_mock,
    )
