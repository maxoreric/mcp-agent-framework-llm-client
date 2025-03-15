"""LLM Client Module for MCP Agent Framework.

This module provides implementations of LLM clients for different API providers.
"""

import json
import os
import requests
from typing import Dict, List, Optional, Any

class LlmClient:
    """
    Base interface for LLM clients.
    """
    def generate_completion(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate a completion for a given prompt."""
        raise NotImplementedError
    
    def generate_chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate a chat response for a given conversation history."""
        raise NotImplementedError

class DeepSeekClient(LlmClient):
    """
    Client for the DeepSeek API.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DeepSeek client.
        
        Args:
            config: Configuration dictionary with the following keys:
                - api_key: DeepSeek API key
                - model: Model to use (e.g., 'deepseek-chat', 'deepseek-reasoner')
                - base_url: Base URL for the API (default: 'https://api.deepseek.com')
                - max_tokens: Maximum number of tokens to generate (default: 2048)
                - temperature: Temperature for sampling (default: 0.7)
                - top_p: Top-p sampling parameter (default: 1.0)
                - frequency_penalty: Frequency penalty (default: 0)
                - presence_penalty: Presence penalty (default: 0)
        """
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("API key is required")
        
        self.model = config.get('model', 'deepseek-chat')
        self.base_url = config.get('base_url', 'https://api.deepseek.com')
        self.max_tokens = config.get('max_tokens', 2048)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 1.0)
        self.frequency_penalty = config.get('frequency_penalty', 0)
        self.presence_penalty = config.get('presence_penalty', 0)
        
        # Additional optional configurations
        self.response_format = config.get('response_format', {"type": "text"})
        self.stream = config.get('stream', False)
        self.stream_options = config.get('stream_options')
        self.stop = config.get('stop')
        self.tools = config.get('tools')
        self.tool_choice = config.get('tool_choice', 'none')
        self.logprobs = config.get('logprobs', False)
        self.top_logprobs = config.get('top_logprobs')
    
    def generate_completion(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a completion using DeepSeek API.
        
        Args:
            prompt: The prompt to generate a completion for
            options: Additional options to override default configuration
            
        Returns:
            The generated text
        """
        # DeepSeek doesn't have a direct text completion endpoint like OpenAI,
        # so we use the chat completion endpoint with a user message
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_chat(messages, options)
        return response.get('content', '')

    def generate_chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate a chat response using DeepSeek API.
        
        Args:
            messages: List of messages in the conversation
            options: Additional options to override default configuration
            
        Returns:
            The assistant's response message
        """
        # Merge default options with provided options
        merged_options = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": self.response_format,
            "stream": self.stream,
            "stream_options": self.stream_options,
            "stop": self.stop,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs
        }
        
        if options:
            merged_options.update(options)
        
        # Prepare the request
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            **merged_options
        }
        
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            # Make the API call
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for non-200 responses
            
            # Parse the response
            result = response.json()
            
            if self.stream:
                # Handle streaming response (basic implementation for MVP)
                return {"role": "assistant", "content": "Streaming responses not implemented in MVP"}
            else:
                # Extract the assistant's message from the response
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    return message
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                
        except requests.RequestException as e:
            raise Exception(f"Error calling DeepSeek API: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"Error parsing DeepSeek API response: {response.text}")

class OpenAiCompatibleClient(LlmClient):
    """
    Client for OpenAI-compatible APIs.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI-compatible client.
        
        Args:
            config: Configuration dictionary with the following keys:
                - api_key: API key
                - model: Model to use
                - base_url: Base URL for the API (default: 'https://api.openai.com/v1')
                - max_tokens: Maximum number of tokens to generate (default: 1000)
                - temperature: Temperature for sampling (default: 0.7)
        """
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("API key is required")
        
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 1.0)
        self.frequency_penalty = config.get('frequency_penalty', 0)
        self.presence_penalty = config.get('presence_penalty', 0)
        
        # Optional configurations
        self.stop = config.get('stop')
        self.tools = config.get('tools')
        self.tool_choice = config.get('tool_choice')
    
    def generate_completion(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a completion using an OpenAI-compatible API.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.generate_chat(messages, options)
        return response.get('content', '')

    def generate_chat(self, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate a chat response using an OpenAI-compatible API.
        """
        # Merge default options with provided options
        merged_options = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }
        
        if options:
            merged_options.update(options)
        
        # Prepare the request
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            **merged_options
        }
        
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            # Make the API call
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the assistant's message from the response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                return message
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except requests.RequestException as e:
            raise Exception(f"Error calling OpenAI-compatible API: {str(e)}")
        except json.JSONDecodeError:
            raise Exception(f"Error parsing API response: {response.text}")

class LlmClientFactory:
    """
    Factory for creating LLM clients.
    """
    @staticmethod
    def get_client(provider: str, config: Dict[str, Any]) -> LlmClient:
        """
        Get an LLM client for the specified provider.
        """
        if provider.lower() == 'deepseek':
            return DeepSeekClient(config)
        elif provider.lower() == 'openai' or provider.lower() == 'openai_compatible':
            return OpenAiCompatibleClient(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

def create_default_llm_client() -> LlmClient:
    """
    Create an LLM client using default configuration from environment variables.
    """
    # Check environment variables for configuration
    provider = os.environ.get('LLM_PROVIDER', 'openai_compatible')
    api_key = os.environ.get('LLM_API_KEY')
    model = os.environ.get('LLM_MODEL')
    base_url = os.environ.get('LLM_BASE_URL')
    
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable must be set")
    
    config = {
        'api_key': api_key
    }
    
    if model:
        config['model'] = model
    
    if base_url:
        config['base_url'] = base_url
    
    # Get other configuration from environment variables
    for key in ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
        env_value = os.environ.get(f'LLM_{key.upper()}')
        if env_value:
            try:
                # Convert string to appropriate type
                if key in ['max_tokens']:
                    config[key] = int(env_value)
                else:
                    config[key] = float(env_value)
            except ValueError:
                # If conversion fails, skip this value
                pass
    
    return LlmClientFactory.get_client(provider, config)