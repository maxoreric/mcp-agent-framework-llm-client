#!/usr/bin/env python
# Example usage of MCP Agent Framework LLM Client

import os
from mcp_agent_framework.llm_client import LlmClientFactory, create_default_llm_client

def deepseek_example():
    """
    Example using DeepSeek API directly
    """
    # Replace with your actual API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
    
    config = {
        'api_key': api_key,
        'model': 'deepseek-chat',
        'temperature': 0.7
    }
    
    client = LlmClientFactory.get_client('deepseek', config)
    
    # Chat example
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is MCP (Model Context Protocol)?"}
    ]
    
    print("\n=== DeepSeek Chat Example ===\n")
    try:
        response = client.generate_chat(messages)
        print(f"Response: {response['content']}")
    except Exception as e:
        print(f"Error: {e}")

def openai_compatible_example():
    """
    Example using OpenAI-compatible API
    """
    # Replace with your actual API key
    api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here")
    
    config = {
        'api_key': api_key,
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7
    }
    
    client = LlmClientFactory.get_client('openai_compatible', config)
    
    print("\n=== OpenAI Compatible Example ===\n")
    try:
        # Simple completion example
        completion = client.generate_completion("What is MCP (Model Context Protocol)?")
        print(f"Completion: {completion}")
    except Exception as e:
        print(f"Error: {e}")

def environment_variables_example():
    """
    Example using environment variables for configuration
    """
    # These would typically be set in the shell or .env file
    # but we're setting them here for demonstration purposes
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["LLM_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
    os.environ["LLM_MODEL"] = "gpt-3.5-turbo"
    os.environ["LLM_TEMPERATURE"] = "0.5"
    
    print("\n=== Environment Variables Example ===\n")
    try:
        client = create_default_llm_client()
        response = client.generate_completion("Explain what MCP is.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("MCP Agent Framework LLM Client Examples")
    print("====================================")
    print("Note: You'll need valid API keys to run these examples.")
    
    # Run the examples
    deepseek_example()
    openai_compatible_example()
    environment_variables_example()
    
    print("\nDone!\n")

if __name__ == "__main__":
    main()