# AxiomAI

[![Hex.pm](https://img.shields.io/hexpm/v/axiom_ai.svg)](https://hex.pm/packages/axiom_ai)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/axiom_ai/)

A unified Elixir client for multiple AI providers.

## Installation

```elixir
def deps do
  [
    {:axiom_ai, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# Vertex AI (Google Cloud)
client = AxiomAi.new(:vertex_ai, %{project_id: "your-gcp-project"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# OpenAI
client = AxiomAi.new(:openai, %{api_key: "your-openai-key"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Anthropic Claude
client = AxiomAi.new(:anthropic, %{api_key: "your-anthropic-key"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Local AI models
client = AxiomAi.new(:local, %{predefined_model: "qwen2.5-0.5b"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Whisper speech-to-text
client = AxiomAi.new(:local, %{predefined_model: "whisper-large-v3-turbo"})
{:ok, response} = AxiomAi.chat(client, "/path/to/audio.wav|Transcribe this audio")
```

## Supported Providers

| Provider | Authentication | Example |
|----------|---------------|---------|
| **Vertex AI** | Service Account, ADC | `AxiomAi.new(:vertex_ai, %{project_id: "my-project"})` |
| **OpenAI** | API Key | `AxiomAi.new(:openai, %{api_key: "sk-..."})` |
| **Anthropic** | API Key | `AxiomAi.new(:anthropic, %{api_key: "sk-..."})` |
| **DeepSeek** | API Key | `AxiomAi.new(:deepseek, %{api_key: "sk-..."})` |
| **AWS Bedrock** | AWS Credentials | `AxiomAi.new(:bedrock, %{model: "anthropic.claude-3-haiku-20240307-v1:0"})` |
| **Local AI** | None | `AxiomAi.new(:local, %{endpoint: "http://localhost:8080"})` |

## Predefined Local Models

```elixir
# List available models
models = AxiomAi.LocalModels.list_models()

# Text generation
"qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b"
"llama3-8b", "mistral-7b", "codellama-7b"

# Speech-to-text
"whisper-large-v3", "whisper-large-v3-turbo"

# OCR
"nanonets-ocr-s"
```

## Authentication

### Vertex AI
```bash
# Local development
gcloud auth application-default login

# Production
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Environment Variables
```bash
# .env file
GOOGLE_CLOUD_PROJECT=your-project-id
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## Configuration Options

```elixir
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project",
  model: "gemini-1.5-pro",      # optional
  region: "us-central1",        # optional
  temperature: 0.7,             # optional
  max_tokens: 1000             # optional
})
```

## License

MIT License