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
| **Local AI** | None | `AxiomAi.new(:local, %{predefined_model: "qwen2.5-0.5b"})` |

## Predefined Local Models

```elixir
# List available models
models = AxiomAi.LocalModels.list_models()
IO.inspect(models)
```

**Available Models:**

**Text Generation:**
- `qwen2.5-0.5b` - Qwen2.5 0.5B
- `qwen2.5-1.5b` - Qwen2.5 1.5B  
- `qwen2.5-3b` - Qwen2.5 3B
- `llama3-8b` - Llama 3 8B
- `mistral-7b` - Mistral 7B
- `codellama-7b` - Code Llama 7B

**Speech-to-Text:**
- `whisper-large-v3` - Whisper Large v3
- `whisper-large-v3-turbo` - Whisper Large v3 Turbo

**OCR:**
- `nanonets-ocr-s` - Nanonets OCR Small

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

## Credentials Configuration

### Vertex AI Service Account Setup

#### Method 1: Service Account File Path
```elixir
config = %{
  project_id: "your-gcp-project",
  service_account_path: "/path/to/service-account.json",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 2: Service Account Key Data
```elixir
# Load credentials from file
{:ok, creds_json} = File.read("/path/to/service-account.json")
{:ok, creds_map} = Jason.decode(creds_json)

config = %{
  project_id: "your-gcp-project",
  service_account_key: creds_map,
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 3: Direct Access Token
```elixir
config = %{
  project_id: "your-gcp-project",
  access_token: "ya29.your-access-token",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 4: Application Default Credentials (ADC)
```elixir
# Automatically detects environment and uses appropriate method:
# - Cloud Run/GCE: Uses metadata service
# - Local: Uses gcloud CLI
config = %{
  project_id: "your-gcp-project",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

### Cloud Run Deployment

**✅ Recommended: Use the default service account**
```elixir
# Cloud Run automatically provides credentials via metadata service
config = %{
  project_id: "your-gcp-project",
  model: "gemini-1.5-pro"
}
```

The library automatically detects Cloud Run environment and uses the metadata service for authentication. No additional configuration needed.

**Alternative: Mount service account file**
```dockerfile
# In your Dockerfile
COPY service-account.json /app/credentials.json
```

```elixir
# In your application
config = %{
  project_id: "your-gcp-project",
  service_account_path: "/app/credentials.json",
  model: "gemini-1.5-pro"
}
```

### Required IAM Permissions
Ensure your service account has the following roles:
- `roles/aiplatform.user` - For Vertex AI API access
- `roles/ml.developer` - For ML model operations (optional)

```bash
# Grant permissions
gcloud projects add-iam-policy-binding your-gcp-project \
  --member="serviceAccount:your-service-account@your-gcp-project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

## License

MIT License