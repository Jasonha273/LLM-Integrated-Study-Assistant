"""
llm_client.py
-------------
Modular wrapper around the Ollama local LLM API.
Handles all HTTP communication, error handling, and response parsing.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60

class LLMClientError(Exception):
    """Raised when the LLM client encounters an error."""
    pass

class LLMClient:
    """
    Clean wrapper around a locally-hosted Ollama LLM.
 
    Usage:
        client = LLMClient(LLMConfig(model="llama3.2"))
        response = client.generate("Explain hash tables in one sentence.")
    """

    def __init__(self, config: Optional[LLMConfig]= None):
        self.config = config or LLMConfig()
        self._endpoint=  f"{self.config.base_url}/api/generate"

    def generate(self,prompt:str, system_prompt: Optional[str] = None)-> str:
        """
        Send a prompt to the LLM and return the text response.
        """
        payload = {
            "model":self.config.model,
            "prompt":prompt,
            "stream":False,
            "options":{
                "temperature":self.config.temperature,
                "num_predict":self.config.max_tokens,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            data= json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type":"application/json"},
                method = "POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise LLMClientError(
                f"Could not connect to Ollama at {self.config.base_url}.\n"
                f"Make sure Ollama is running: `ollama serve`\nDetails: {e}"
            ) from e
        except (KeyError, json.JSONDecodeError) as e:
            raise LLMClientError(f"Unexpected response format from LLM: {e}") from e

    def is_available(self)-> bool:
        """Checks if the Ollama Server is available"""
        try:
            url = f"{self.config.base_url}/api/tags"
            urllib.request.urlopen(url,timeout=5)
            return True
        except Exception:
            return False
        